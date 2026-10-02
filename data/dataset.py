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
        img_size: int = 256,
    ):
        """
        Args:
            lerobot_dataset: Loaded LeRobotDataset instance
            normalizer: Normalizer loaded with action_stats.json
            action_horizon: Number of future action steps to chunk (default: 16)
            target_tasks: List of task names/substrings to include (default: 3 PoC tasks)
            shots_per_task: If specified, limit the dataset to N episodes per task (e.g. 5 or 10)
            img_size: Image spatial resolution for SigLIP vision backbone (default: 256)
        """
        self.dataset = lerobot_dataset
        self.normalizer = normalizer
        self.action_horizon = action_horizon
        self.img_size = img_size
        self.target_tasks = target_tasks or [
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
            "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
            "push_the_plate_to_the_front_of_the_stove",
        ]

        # 1. Index valid episodes matching target tasks
        self.valid_episodes = self._filter_episodes(shots_per_task)
        # 2. Build frame index mapping: dataset_idx -> (episode_id, frame_in_episode)
        self.valid_frames = self._build_frame_indices()

    def _filter_episodes(self, shots_per_task: Optional[int]) -> List[int]:
        """Filters dataset episodes for the target tasks, with optional N-shot quota."""
        task_counts: Dict[str, int] = {t: 0 for t in self.target_tasks}
        selected_episodes: List[int] = []

        num_episodes = self.dataset.num_episodes
        for ep in range(num_episodes):
            start_idx = self.dataset.episode_data_index["from"][ep].item()
            task_str = self.dataset[start_idx].get("task", "")
            task_str_clean = task_str.replace(" ", "_").lower()

            for target in self.target_tasks:
                target_clean = target.lower()
                target_words = target.replace("_", " ").lower()

                if target_clean in task_str_clean or target_words in task_str.lower():
                    if shots_per_task is None or task_counts[target] < shots_per_task:
                        task_counts[target] += 1
                        selected_episodes.append(ep)
                        break

        print(f"Filtered {len(selected_episodes)} episodes matching targets across {num_episodes} total episodes.")
        for t, count in task_counts.items():
            print(f"  - Task '{t[:40]}...': {count} episodes")
        return selected_episodes

    def _build_frame_indices(self) -> List[Tuple[int, int, int]]:
        """
        Builds a list of tuples: (global_frame_idx, episode_idx, episode_end_idx).
        This allows O(1) lookup during dataset indexing.
        """
        frames = []
        for ep in self.valid_episodes:
            ep_start = self.dataset.episode_data_index["from"][ep].item()
            ep_end = self.dataset.episode_data_index["to"][ep].item()  # inclusive or exclusive
            for idx in range(ep_start, ep_end):
                frames.append((idx, ep, ep_end))
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
