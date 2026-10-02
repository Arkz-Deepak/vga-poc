"""
VGA Dataset & Action Chunking Pipeline for LIBERO-Spatial.

Mathematical and Engineering Foundations:
1. Action Chunking:
   Robotic policies operating step-by-step accumulate compound errors and high latency.
   We predict an action horizon of H = 16 steps:
       A_{t:t+H} = [a_t, a_{t+1}, ..., a_{t+H-1}] in R^{16 x 7}
   where each action vector a_i contains:
       [dx, dy, dz, rx, ry, rz, gripper]
   - dx, dy, dz: 3D delta end-effector position in meters.
   - rx, ry, rz: 3D delta rotation represented in axis-angle coordinates.
   - gripper: Gripper state normalized in [0, 1] or binary {-1, 1}.

2. Empirical Normalization:
   Continuous diffusion/flow-matching models train optimally when target distributions
   have zero mean and unit variance:
       a_norm = (a - mu_d) / sigma_d
   We load empirical (mu_d, sigma_d) from `configs/action_stats.json`.

3. Task Filtering (5-Shot / 10-Shot Subsetting):
   Our PoC benchmarks against 3 LIBERO-Spatial tasks under extreme sample efficiency:
   5 demonstration episodes and 10 demonstration episodes.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Normalizer:
    """Handles bidirectional normalization of 7D action vectors using empirical statistics."""

    def __init__(self, stats_path: Union[str, Path]):
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Action statistics file not found at: {stats_path}")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.mu_d = torch.tensor(stats["mu_d"], dtype=torch.float32)
        self.sigma_d = torch.tensor(stats["sigma_d"], dtype=torch.float32)
        # Ensure non-zero standard deviation to prevent division by zero
        self.sigma_d = torch.clamp(self.sigma_d, min=1e-5)
        self.sigma_pos_sq = stats.get("sigma_pos_sq", 1.0)
        self.sigma_rot_sq = stats.get("sigma_rot_sq", 1.0)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Normalize actions from physical space to standard normal space.
        Args:
            actions: Tensor of shape (..., 7)
        Returns:
            normalized_actions: Tensor of shape (..., 7) with mean ~0, std ~1
        """
        mu = self.mu_d.to(device=actions.device, dtype=actions.dtype)
        sigma = self.sigma_d.to(device=actions.device, dtype=actions.dtype)
        return (actions - mu) / sigma

    def unnormalize(self, norm_actions: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize actions back to physical robotic action units.
        Args:
            norm_actions: Tensor of shape (..., 7) in normalized space
        Returns:
            physical_actions: Tensor of shape (..., 7)
        """
        mu = self.mu_d.to(device=norm_actions.device, dtype=norm_actions.dtype)
        sigma = self.sigma_d.to(device=norm_actions.device, dtype=norm_actions.dtype)
        return norm_actions * sigma + mu


class LiberoSpatialDataset(Dataset):
    """
    Dataset wrapper for LIBERO-Spatial episodes providing 16-step action chunking,
    image observation preprocessing, and task filtering.
    """

    def __init__(
        self,
        lerobot_dataset,
        normalizer: Normalizer,
        action_horizon: int = 16,
        target_tasks: Optional[List[str]] = None,
        shots_per_task: Optional[int] = None,
        skip_per_task: int = 0,
        img_size: int = 256,
    ):
        """
        Args:
            lerobot_dataset: Loaded LeRobotDataset instance
            normalizer: Normalizer loaded with action_stats.json
            action_horizon: Number of future action steps to chunk (default: 16)
            target_tasks: List of task names/substrings to include (default: 3 PoC tasks)
            shots_per_task: If specified, limit the dataset to N episodes per task (e.g. 5 or 10)
            skip_per_task: Number of initial episodes per task to skip (useful for held-out test splits)
            img_size: Image spatial resolution for SigLIP vision backbone (default: 256)
        """
        self.dataset = lerobot_dataset
        self.normalizer = normalizer
        self.action_horizon = action_horizon
        self.img_size = img_size
        self.skip_per_task = skip_per_task
        self.target_tasks = target_tasks or [
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
            "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
            "push_the_plate_to_the_front_of_the_stove",
        ]

        # 1. Index valid episodes matching target tasks
        self.valid_episodes = self._filter_episodes(shots_per_task, skip_per_task)
        # 2. Build frame index mapping: dataset_idx -> (episode_id, frame_in_episode)
        self.valid_frames = self._build_frame_indices()

    def _get_all_episodes_metadata(self) -> List[Dict]:
        """
        Universally extracts (ep_idx, from, to, task) across all LeRobot versions:
        - Modern LeRobot (v0.6+ / v3.0): dataset.meta.episodes
        - Legacy LeRobot (v2.0 / v2.1): dataset.episode_data_index
        - Fallback: dynamic discovery
        """
        episodes_info = []

        # 1. Modern LeRobot (v0.6+ / v3.0) via metadata parquet
        if hasattr(self.dataset, "meta") and hasattr(self.dataset.meta, "episodes") and self.dataset.meta.episodes is not None:
            ep_meta = self.dataset.meta.episodes
            num_ep = len(ep_meta)
            cols = ep_meta.column_names if hasattr(ep_meta, "column_names") else list(ep_meta.features.keys()) if hasattr(ep_meta, "features") else []
            from_col = "dataset_from_index" if "dataset_from_index" in cols else None
            to_col = "dataset_to_index" if "dataset_to_index" in cols else None
            tasks_col = "tasks" if "tasks" in cols else None

            if from_col and to_col:
                from_list = ep_meta[from_col]
                to_list = ep_meta[to_col]
                tasks_list = ep_meta[tasks_col] if tasks_col else [None] * num_ep

                for ep in range(num_ep):
                    t_val = tasks_list[ep]
                    if t_val is not None and isinstance(t_val, (list, np.ndarray)) and len(t_val) > 0:
                        task_name = str(t_val[0])
                    else:
                        task_name = str(t_val or "")
                    episodes_info.append({
                        "ep_idx": ep,
                        "from": int(from_list[ep]),
                        "to": int(to_list[ep]),
                        "task": task_name,
                    })
                return episodes_info

        # 2. Legacy LeRobot (v2.0 / v2.1)
        if hasattr(self.dataset, "episode_data_index"):
            from_tensor = self.dataset.episode_data_index["from"]
            to_tensor = self.dataset.episode_data_index["to"]
            for ep in range(len(from_tensor)):
                f_idx = from_tensor[ep].item()
                t_idx = to_tensor[ep].item()
                task_name = self.dataset[f_idx].get("task", "")
                episodes_info.append({
                    "ep_idx": ep,
                    "from": f_idx,
                    "to": t_idx,
                    "task": str(task_name),
                })
            return episodes_info

        # 3. Fallback: single episode covering all frames
        total_len = len(self.dataset)
        episodes_info.append({
            "ep_idx": 0,
            "from": 0,
            "to": total_len,
            "task": str(self.dataset[0].get("task", "")),
        })
        return episodes_info

    def _filter_episodes(self, shots_per_task: Optional[int], skip_per_task: int = 0) -> List[Dict]:
        """Filters dataset episodes for the target tasks, with optional N-shot quota and initial skip offset."""
        all_eps = self._get_all_episodes_metadata()
        task_seen: Dict[str, int] = {t: 0 for t in self.target_tasks}
        task_counts: Dict[str, int] = {t: 0 for t in self.target_tasks}
        selected: List[Dict] = []

        for ep_info in all_eps:
            task_str = ep_info["task"]
            task_clean = task_str.replace(" ", "_").lower()

            matched_target = None
            for target in self.target_tasks:
                target_clean = target.lower()
                target_words = target.replace("_", " ").lower()

                if (target_clean in task_clean or 
                    target_words in task_str.lower() or 
                    any(word in task_str.lower() for word in ["black bowl", "stove", "ramekin", "basket"])):
                    matched_target = target
                    break

            if matched_target is not None:
                task_seen[matched_target] += 1
                if task_seen[matched_target] > skip_per_task:
                    if shots_per_task is None or task_counts[matched_target] < shots_per_task:
                        task_counts[matched_target] += 1
                        selected.append(ep_info)

        # Fallback if no task matched exactly
        if len(selected) == 0:
            offset = skip_per_task * len(self.target_tasks)
            limit = min(shots_per_task * len(self.target_tasks) if shots_per_task else 15, len(all_eps) - offset)
            selected = all_eps[offset : offset + limit]

        print(f"Filtered {len(selected)} episodes matching targets across {len(all_eps)} total episodes (skipped initial {skip_per_task}/task).")
        return selected

    def _build_frame_indices(self) -> List[Tuple[int, int, int]]:
        """
        Builds a list of tuples: (global_frame_idx, episode_idx, episode_end_idx).
        This allows O(1) lookup during dataset indexing.
        """
        frames = []
        for ep_info in self.valid_episodes:
            f_idx = ep_info["from"]
            t_idx = ep_info["to"]
            ep_id = ep_info["ep_idx"]
            for idx in range(f_idx, t_idx):
                frames.append((idx, ep_id, t_idx))
        print(f"Total valid training frames indexed: {len(frames):,}")
        return frames

    def __len__(self) -> int:
        return len(self.valid_frames)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves observation at time t and the H-step future action chunk A_{t:t+H}.
        """
        frame_idx, ep_idx, ep_end = self.valid_frames[index]
        current_sample = self.dataset[frame_idx]

        # 1. Process RGB observations (front camera and wrist camera)
        img_front = current_sample.get("observation.images.image", None)
        img_wrist = current_sample.get("observation.images.wrist_image", None)

        if img_front is None:
            # Fallback for alternative key naming
            for k in current_sample:
                if "image" in k and "wrist" not in k:
                    img_front = current_sample[k]
                    break
        if img_wrist is None:
            for k in current_sample:
                if "wrist" in k:
                    img_wrist = current_sample[k]
                    break

        # Resize images to standard visual backbone resolution (C, H, W)
        def process_img(img_tensor):
            if img_tensor is None:
                return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            if img_tensor.dtype == torch.uint8:
                img_tensor = img_tensor.float() / 255.0
            # Interpolate to target size if needed
            if img_tensor.shape[-2:] != (self.img_size, self.img_size):
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            # Map pixel values to [-1.0, 1.0] for SigLIP
            return img_tensor * 2.0 - 1.0

        image_front_proc = process_img(img_front)
        image_wrist_proc = process_img(img_wrist)

        # 2. Extract Robot Proprioceptive State (joint angles / gripper / end-effector pos)
        state = current_sample.get("observation.state", torch.zeros(8, dtype=torch.float32))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # 3. Assemble H = 16 Future Action Chunk
        # If the episode ends before t + H, pad with the terminal action
        action_seq = []
        for step in range(self.action_horizon):
            target_idx = min(frame_idx + step, ep_end - 1)
            raw_action = self.dataset[target_idx]["action"]
            if isinstance(raw_action, np.ndarray):
                raw_action = torch.from_numpy(raw_action).float()
            elif not isinstance(raw_action, torch.Tensor):
                raw_action = torch.tensor(raw_action, dtype=torch.float32)
            action_seq.append(raw_action)

        action_chunk = torch.stack(action_seq, dim=0)  # Shape: [16, 7]

        # 4. Normalize Action Chunk
        normalized_actions = self.normalizer.normalize(action_chunk)

        # 5. Extract Language Instruction string
        task_instruction = current_sample.get(
            "task",
            "pick up the black bowl between the plate and the ramekin and place it on the plate",
        )

        return {
            "image_front": image_front_proc,        # [3, 256, 256]
            "image_wrist": image_wrist_proc,        # [3, 256, 256]
            "state": state,                         # [state_dim]
            "actions": normalized_actions,          # [16, 7] normalized
            "raw_actions": action_chunk,            # [16, 7] physical units
            "task": task_instruction,              # string
        }
