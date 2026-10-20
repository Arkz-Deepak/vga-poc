"""
data/dataset.py
Trajectory Chunk Ingestion and Normalization Dataset for LIBERO-Spatial.

Key Capabilities:
1. Robust Multi-Key Discovery: Automatically searches for observation keys across LIBERO
   variants ('agentview_rgb', 'robot0_eye_in_hand_rgb', 'robot0_eef_pos', 'robot0_eef_quat',
   'robot0_gripper_qpos', etc.).
2. Relative SE(3) Wrist Pose Construction: Computes relative wrist rotation R_rel(t) = R_eef(t) @ R_eef(0)^T
   so that nominal downward home orientation equals Identity I_3 (avoiding 180-deg optical frame flips)
   while accurately rotating 3D Ray-RoPE rays whenever the wrist pitches or rolls at workspace edges.
3. Physical Proprioceptive Grasp State: Uses physical finger encoder separation ('robot0_gripper_qpos')
   alongside commands so a_prev[:, 6] reflects true physical object capture in [-1.0, +1.0].
4. Dual-Camera & Decoupled Action Targets: Provides both normalized 6-DoF flow targets and raw binary
   gripper targets for decoupled Flow-Matching + BCE supervision.
"""
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from configs.poc_config import CONFIG


def quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Converts a Robosuite unit quaternion [x, y, z, w] into a 3x3 rotation matrix R in SO(3).
    """
    q = np.asarray(quat, dtype=np.float64).flatten()
    if q.shape[0] != 4:
        return np.eye(3, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / norm

    R = np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float32)
    return R


def axis_angle_to_rotmat(vec: np.ndarray) -> np.ndarray:
    """
    Converts a 3D axis-angle orientation vector (Rodrigues) into a 3x3 rotation matrix R in SO(3).
    """
    v = np.asarray(vec, dtype=np.float64).flatten()
    if v.shape[0] != 3:
        return np.eye(3, dtype=np.float32)
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = v / theta
    kx, ky, kz = k[0], k[1], k[2]
    K = np.array([
        [0.0, -kz,  ky],
        [kz,  0.0, -kx],
        [-ky, kx,  0.0],
    ], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    return R.astype(np.float32)


def compute_relative_wrist_rotmat(
    quat_t: Optional[np.ndarray],
    quat_0: Optional[np.ndarray],
    ori_t: Optional[np.ndarray] = None,
    ori_0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes the relative wrist rotation matrix R_rel(t) = R_eef(t) @ R_eef(0)^T.
    At nominal downward home pose (t=0), R_rel(0) = I_3, preserving the Ray-RoPE optical axis
    convention while capturing exact 3D wrist pitch/roll deviations across the workspace.
    """
    if quat_t is not None and quat_0 is not None:
        r_t = quat_xyzw_to_rotmat(quat_t)
        r_0 = quat_xyzw_to_rotmat(quat_0)
        return (r_t @ r_0.T).astype(np.float32)
    if ori_t is not None and ori_0 is not None:
        r_t = axis_angle_to_rotmat(ori_t)
        r_0 = axis_angle_to_rotmat(ori_0)
        return (r_t @ r_0.T).astype(np.float32)
    return np.eye(3, dtype=np.float32)


def compute_physical_gripper_state(
    cmd_grip: np.ndarray,
    gripper_qpos: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Maps physical finger aperture width w = |qpos[0] - qpos[1]| and command history into
    an in-distribution [-1.0 (open / empty air), +1.0 (holding object)] state signal.
    """
    cmd_sign = np.where(cmd_grip > 0.0, 1.0, -1.0).astype(np.float32)
    if gripper_qpos is None:
        return cmd_sign

    q = np.asarray(gripper_qpos, dtype=np.float32)
    if q.ndim == 2 and q.shape[-1] >= 2:
        aperture = np.abs(q[:, 0] - q[:, 1])
    elif q.ndim == 2 and q.shape[-1] == 1:
        aperture = np.abs(q[:, 0]) * 2.0
    else:
        return cmd_sign

    # Physical grasp requires commanded closure AND non-zero finger separation (0.0025m < w < 0.068m)
    # If fingers pinch completely shut on empty air (w <= 0.0025m) or are wide open (w >= 0.068m), state is -1.0
    holding_obj = (cmd_grip > 0.0) & (aperture > 0.0025) & (aperture < 0.068)
    return np.where(holding_obj, 1.0, -1.0).astype(np.float32)


from pathlib import Path


class Normalizer:
    """Handles bidirectional normalization of 7D action vectors using empirical statistics."""

    def __init__(self, stats_path: Optional[Union[str, Path]] = None):
        target_path: Optional[Path] = None
        repo_root = Path(__file__).resolve().parent.parent

        candidate_paths = []
        if stats_path is not None:
            candidate_paths.append(Path(stats_path))
            candidate_paths.append(repo_root / stats_path)
            candidate_paths.append(repo_root / "configs" / Path(stats_path).name)

        candidate_paths.extend([
            repo_root / "configs" / "action_stats.json",
            repo_root / "configs" / "action_stats_10shot.json",
            Path("configs/action_stats.json"),
            Path("configs/action_stats_10shot.json"),
            Path("/kaggle/working/vga-poc/configs/action_stats.json"),
            Path("/kaggle/working/vga-poc/configs/action_stats_10shot.json"),
            Path("/kaggle/working/configs/action_stats.json"),
        ])

        for p in candidate_paths:
            try:
                if p.is_file():
                    target_path = p
                    break
            except Exception:
                pass

        if target_path is not None and target_path.exists():
            with open(target_path, "r") as f:
                stats = json.load(f)
            mean_key = "mu" if "mu" in stats else ("mu_d" if "mu_d" in stats else "mean")
            std_key = "sigma" if "sigma" in stats else ("sigma_d" if "sigma_d" in stats else "std")
            self.mu_d = torch.tensor(stats[mean_key], dtype=torch.float32)
            self.sigma_d = torch.tensor(stats[std_key], dtype=torch.float32)
            self.sigma_d = torch.clamp(self.sigma_d, min=1e-5)
            self.sigma_pos_sq = stats.get("sigma_pos_sq", 0.18988)
            self.sigma_rot_sq = stats.get("sigma_rot_sq", 0.00321)
            self.stats_path = str(target_path)
        else:
            self.mu_d = torch.tensor([0.172505, 0.101684, -0.144693, 0.009553, 0.000129, -0.032239, -0.012093], dtype=torch.float32)
            self.sigma_d = torch.tensor([0.507023, 0.206400, 0.616726, 0.043281, 0.075032, 0.068463, 0.999927], dtype=torch.float32)
            self.sigma_d = torch.clamp(self.sigma_d, min=1e-5)
            self.sigma_pos_sq = 0.22667457
            self.sigma_rot_sq = 0.00406339
            self.stats_path = "default_empirical"

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        mu = self.mu_d.to(device=actions.device, dtype=actions.dtype)
        sigma = self.sigma_d.to(device=actions.device, dtype=actions.dtype)
        return (actions - mu) / sigma

    def unnormalize(self, norm_actions: torch.Tensor) -> torch.Tensor:
        mu = self.mu_d.to(device=norm_actions.device, dtype=norm_actions.dtype)
        sigma = self.sigma_d.to(device=norm_actions.device, dtype=norm_actions.dtype)
        return norm_actions * sigma + mu



class LiberoSpatialDataset(Dataset):
    def __init__(
        self,
        episodes: List[Dict[str, Any]],
        indices: List[Tuple[int, int]],
        chunk_horizon: int = getattr(CONFIG, "chunk_horizon", 16),
        prefix_len: int = getattr(CONFIG, "buffer_prefix_len", 4),
        action_stats: Optional[Dict[str, Any]] = None
    ):
        self.episodes = episodes
        self.indices = indices
        self.chunk_horizon = chunk_horizon
        self.prefix_len = prefix_len

        # Default static camera translation in Franka base frame
        self.t_base_cam_default = np.array([0.0, -0.25, 0.60], dtype=np.float32)

        # Action normalization parameters
        if action_stats is not None:
            mean_key = "mean" if "mean" in action_stats else "mu"
            std_key = "std" if "std" in action_stats else "sigma"
            self.mu = torch.as_tensor(action_stats[mean_key], dtype=torch.float32)
            self.sigma = torch.as_tensor(action_stats[std_key], dtype=torch.float32)
        else:
            self.mu = torch.zeros(getattr(CONFIG, "action_dim", 7), dtype=torch.float32)
            self.sigma = torch.ones(getattr(CONFIG, "action_dim", 7), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, t = self.indices[idx]
        ep = self.episodes[ep_idx]

        # 1. Front Third-Person Camera Observation
        front_rgb = ep["front_rgb"][t]  # [H, W, C], uint8
        rgb = torch.from_numpy(front_rgb).permute(2, 0, 1).float() / 255.0

        # 2. Wrist Eye-in-Hand Camera Observation
        if ep["wrist_rgb"] is not None and len(ep["wrist_rgb"]) > t:
            wrist_rgb = ep["wrist_rgb"][t]
            rgb_wrist = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float() / 255.0
        else:
            rgb_wrist = rgb.clone()

        # 3. Relative SE(3) End-Effector / Wrist Pose (Position + Relative Orientation from Home)
        if ep.get("eef_pos") is not None and len(ep["eef_pos"]) > t:
            t_wrist_np = ep["eef_pos"][t]
        else:
            t_wrist_np = np.array([0.0, 0.0, 0.90], dtype=np.float32)
        t_base_wrist = torch.from_numpy(t_wrist_np).float()

        quat_t = ep["eef_quat"][t] if (ep.get("eef_quat") is not None and len(ep["eef_quat"]) > t) else None
        quat_0 = ep["eef_quat"][0] if (ep.get("eef_quat") is not None and len(ep["eef_quat"]) > 0) else None
        ori_t = ep["eef_ori"][t] if (ep.get("eef_ori") is not None and len(ep["eef_ori"]) > t) else None
        ori_0 = ep["eef_ori"][0] if (ep.get("eef_ori") is not None and len(ep["eef_ori"]) > 0) else None

        r_wrist_np = compute_relative_wrist_rotmat(quat_t, quat_0, ori_t, ori_0)

        t_base_wrist_mat = torch.eye(4, dtype=torch.float32)
        t_base_wrist_mat[:3, :3] = torch.from_numpy(r_wrist_np).float()
        t_base_wrist_mat[:3, 3] = t_base_wrist

        # Static front camera pose
        t_base_cam_mat = torch.eye(4, dtype=torch.float32)
        t_base_cam_mat[:3, 3] = torch.from_numpy(self.t_base_cam_default).float()

        # 4. Target Action Chunk [chunk_horizon, 7]
        actions_raw = ep["actions"][t : t + self.chunk_horizon]
        if len(actions_raw) < self.chunk_horizon:
            pad_count = self.chunk_horizon - len(actions_raw)
            last_act = actions_raw[-1:] if len(actions_raw) > 0 else np.zeros((1, 7), dtype=np.float32)
            pad = np.repeat(last_act, pad_count, axis=0)
            actions_raw = np.vstack([actions_raw, pad])

        actions_tensor = torch.from_numpy(actions_raw).float()
        actions_norm = (actions_tensor - self.mu) / (self.sigma + 1e-8)
        if actions_tensor.shape[-1] >= 7:
            actions_norm[:, 6] = actions_tensor[:, 6]

        # 5. Past Action History Buffer [prefix_len, 7] with Physical Proprioceptive Gripper Channel
        if t >= self.prefix_len:
            hist_raw = ep["actions"][t - self.prefix_len : t, :7].copy()
            qpos_slice = ep["gripper_qpos"][t - self.prefix_len : t] if ep.get("gripper_qpos") is not None else None
        else:
            pad_len = self.prefix_len - t
            start_act = ep["actions"][0, :7]
            pad = np.repeat(start_act[np.newaxis, :], pad_len, axis=0)
            if t > 0:
                hist_raw = np.vstack([pad, ep["actions"][:t, :7]]).copy()
            else:
                hist_raw = pad.copy()
            if ep.get("gripper_qpos") is not None:
                start_q = ep["gripper_qpos"][0:1]
                q_pad = np.repeat(start_q, pad_len, axis=0)
                qpos_slice = np.vstack([q_pad, ep["gripper_qpos"][:t]]) if t > 0 else q_pad
            else:
                qpos_slice = None

        a_prev_raw = torch.from_numpy(hist_raw).float()
        a_prev = (a_prev_raw - self.mu) / (self.sigma + 1e-8)
        if a_prev_raw.shape[-1] >= 7:
            phys_grip = compute_physical_gripper_state(hist_raw[:, 6], qpos_slice)
            a_prev[:, 6] = torch.from_numpy(phys_grip).float()

        # 6. Extracted Camera Intrinsics
        cam_intrinsics = None
        if ep.get("front_intrinsics") is not None:
            cam_intrinsics = torch.from_numpy(ep["front_intrinsics"]).float()

        wrist_intrinsics = None
        if ep.get("wrist_intrinsics") is not None:
            wrist_intrinsics = torch.from_numpy(ep["wrist_intrinsics"]).float()

        # 7. Extracted Spatial Forcing 3D Features
        target_3d_features = None
        if ep.get("spatial_features") is not None and len(ep["spatial_features"]) > t:
            target_3d_features = torch.from_numpy(ep["spatial_features"][t]).float()

        batch_dict = {
            "rgb": rgb,
            "rgb_wrist": rgb_wrist,
            "t_base_cam": t_base_cam_mat,
            "t_base_wrist": t_base_wrist_mat,
            "actions_norm": actions_norm,
            "actions_raw": actions_tensor,
            "a_prev": a_prev,
            "prompt": ep["prompt"]
        }

        if cam_intrinsics is not None:
            batch_dict["cam_intrinsics"] = cam_intrinsics
        if wrist_intrinsics is not None:
            batch_dict["wrist_intrinsics"] = wrist_intrinsics
        if target_3d_features is not None:
            batch_dict["target_3d_features"] = target_3d_features

        return batch_dict

    @staticmethod
    def compute_action_stats(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates global mean and standard deviation normalizers across demonstrations."""
        all_actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)
        mu = np.mean(all_actions, axis=0).astype(np.float32)
        sigma = np.std(all_actions, axis=0).astype(np.float32)
        sigma = np.clip(sigma, 1e-4, None)

        if len(mu) >= 7:
            mu[6] = 0.0
            sigma[6] = 1.0

        sigma_pos_sq = float(np.var(all_actions[:, :3]))
        sigma_rot_sq = float(np.var(all_actions[:, 3:6]))

        return {
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "mean": mu.tolist(),
            "std": sigma.tolist(),
            "sigma_pos_sq": sigma_pos_sq,
            "sigma_rot_sq": sigma_rot_sq
        }

    @classmethod
    def from_disk(
        cls,
        data_path: str,
        num_shots: Optional[int] = 5,
        chunk_horizon: int = getattr(CONFIG, "chunk_horizon", 16),
        prefix_len: int = getattr(CONFIG, "buffer_prefix_len", 4),
        action_stats: Optional[Dict[str, Any]] = None,
        seed: int = 0
    ) -> "LiberoSpatialDataset":
        """Loads and indexes demonstration trajectories from HDF5/NPZ archives."""
        if os.path.isfile(data_path):
            files = [data_path]
        else:
            files = sorted(glob.glob(os.path.join(data_path, "**", "*.hdf5"), recursive=True))
            if not files:
                files = sorted(glob.glob(os.path.join(data_path, "**", "*.h5"), recursive=True))
            if not files:
                files = sorted(glob.glob(os.path.join(data_path, "**", "*.npz"), recursive=True))

        if not files:
            raise FileNotFoundError(f"No demonstration files found under '{data_path}'")

        episodes: List[Dict[str, Any]] = []

        for file_path in files:
            base_name = os.path.basename(file_path).replace("_demo.hdf5", "").replace(".hdf5", "").replace(".npz", "")
            default_prompt = base_name.replace("_", " ")

            if file_path.endswith(".npz"):
                with np.load(file_path, allow_pickle=True) as f:
                    episodes.append({
                        "front_rgb": f["rgb"],
                        "wrist_rgb": None,
                        "depth": f.get("depth", None),
                        "eef_pos": None,
                        "eef_quat": None,
                        "eef_ori": None,
                        "gripper_qpos": None,
                        "actions": f["actions"].astype(np.float32),
                        "prompt": str(f.get("prompt", default_prompt)),
                        "front_intrinsics": None,
                        "wrist_intrinsics": None,
                        "spatial_features": None
                    })
                continue

            with h5py.File(file_path, "r") as f:
                if "data" not in f:
                    continue
                demo_keys = sorted(list(f["data"].keys()))

                if num_shots is not None and len(demo_keys) > num_shots:
                    selected_keys = demo_keys[:num_shots]
                else:
                    selected_keys = demo_keys

                for key in selected_keys:
                    demo = f["data"][key]
                    if "actions" not in demo:
                        continue

                    actions = demo["actions"][:]
                    obs = demo.get("obs", demo)

                    # 1. Extract Front RGB
                    front_rgb = None
                    for fk in ["agentview_rgb", "agentview_image", "agentview", "front_rgb"]:
                        if fk in obs:
                            front_rgb = obs[fk][:]
                            break
                    if front_rgb is None:
                        continue

                    # 2. Extract Wrist RGB
                    wrist_rgb = None
                    for wk in ["robot0_eye_in_hand_rgb", "robot0_eye_in_hand_image", "eye_in_hand_rgb", "wrist_rgb"]:
                        if wk in obs:
                            wrist_rgb = obs[wk][:]
                            break
                    if wrist_rgb is None:
                        wrist_rgb = front_rgb

                    depth = None
                    for dk in ["agentview_depth", "depth"]:
                        if dk in obs:
                            depth = obs[dk][:]
                            break

                    # 3. Extract End-Effector Position, Orientation & Physical Gripper Qpos
                    eef_pos = None
                    for ek in ["robot0_eef_pos", "ee_pos", "eef_pos"]:
                        if ek in obs:
                            eef_pos = obs[ek][:]
                            break
                        elif ek in demo:
                            eef_pos = demo[ek][:]
                            break

                    if eef_pos is None and "ee_states" in obs:
                        eef_pos = obs["ee_states"][:, :3]

                    if eef_pos is None:
                        T_len = len(actions)
                        eef_pos = np.zeros((T_len, 3), dtype=np.float32)
                        nominal_pos = np.array([0.0, 0.0, 0.85], dtype=np.float32)
                        eef_pos[:] = nominal_pos + np.cumsum(actions[:, :3] * 0.05, axis=0)

                    eef_quat = None
                    for qk in ["robot0_eef_quat", "eef_quat", "ee_quat"]:
                        if qk in obs:
                            eef_quat = obs[qk][:]
                            break
                        elif qk in demo:
                            eef_quat = demo[qk][:]
                            break

                    eef_ori = None
                    for ok in ["ee_ori", "eef_ori", "robot0_eef_ori"]:
                        if ok in obs:
                            eef_ori = obs[ok][:]
                            break
                        elif ok in demo:
                            eef_ori = demo[ok][:]
                            break
                    if eef_ori is None and "ee_states" in obs and obs["ee_states"].shape[-1] >= 6:
                        eef_ori = obs["ee_states"][:, 3:6]

                    gripper_qpos = None
                    for gk in ["robot0_gripper_qpos", "gripper_qpos", "gripper_states"]:
                        if gk in obs:
                            gripper_qpos = obs[gk][:]
                            break
                        elif gk in demo:
                            gripper_qpos = demo[gk][:]
                            break

                    # 4. Extract Camera Intrinsics
                    front_intrinsics = None
                    if "agentview_intrinsics" in obs:
                        front_intrinsics = obs["agentview_intrinsics"][:]

                    wrist_intrinsics = None
                    if "robot0_eye_in_hand_intrinsics" in obs:
                        wrist_intrinsics = obs["robot0_eye_in_hand_intrinsics"][:]

                    # 5. Extract Offline 3D Spatial Features
                    spatial_features = None
                    for sk in ["target_3d_features", "spatial_features"]:
                        if sk in obs:
                            spatial_features = obs[sk][:]
                            break

                    # 6. Extract Prompt
                    prompt = default_prompt
                    if "problem_info" in f["data"].attrs:
                        try:
                            info = f["data"].attrs["problem_info"]
                            if isinstance(info, str) and "language_instruction" in info:
                                prompt = json.loads(info).get("language_instruction", prompt)
                        except Exception:
                            pass

                    episodes.append({
                        "front_rgb": front_rgb,
                        "wrist_rgb": wrist_rgb,
                        "depth": depth,
                        "eef_pos": eef_pos,
                        "eef_quat": eef_quat,
                        "eef_ori": eef_ori,
                        "gripper_qpos": gripper_qpos,
                        "actions": actions.astype(np.float32),
                        "prompt": prompt,
                        "front_intrinsics": front_intrinsics,
                        "wrist_intrinsics": wrist_intrinsics,
                        "spatial_features": spatial_features
                    })

        indices: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            T = len(ep["actions"])
            if T <= chunk_horizon:
                indices.append((ep_idx, 0))
            else:
                for t in range(0, T - chunk_horizon + 1):
                    indices.append((ep_idx, t))

        return cls(
            episodes=episodes,
            indices=indices,
            chunk_horizon=chunk_horizon,
            prefix_len=prefix_len,
            action_stats=action_stats
        )