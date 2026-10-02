"""
Affine Schmitt Trigger Gripper Controller with Hysteresis.

Role in VGA Architecture:
-------------------------
Continuous diffusion models predict real-valued gripper values in range [-1.0, 1.0].
However, physical robotic grippers are binary or thresholded (Open or Closed).
Directly applying a hard threshold at 0.0 causes "gripper chatter" - if the model's
output oscillates slightly between +0.01 and -0.01 due to sensor noise, the gripper
repeatedly opens and closes at 50 Hz, dropping whatever object it was holding!

Solution:
---------
1. Affine normalization: maps continuous [-1, 1] to normalized [0, 1]:
   g_bar = (g_hat + 1) / 2
2. Schmitt trigger with hysteresis deadband:
   - High threshold (0.65): only transitions to CLOSED (1.0) when g_bar > 0.65.
   - Low threshold (0.35): only transitions to OPEN (0.0) when g_bar < 0.35.
   - In between (0.35 <= g_bar <= 0.65): retains its previous state (g_prev).
This guarantees stable, chatter-free grasping.
"""

from typing import Union
import torch


class SchmittTriggerGripper:
    """
    Stateful gripper controller implementing affine rescaling and hysteresis deadband filtering.
    """
    def __init__(
        self,
        low_thresh: float = 0.45,
        high_thresh: float = 0.55,
        open_val: float = -1.0,
        close_val: float = 1.0,
        initial_state: float = None,
    ):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.open_val = float(open_val)
        self.close_val = float(close_val)
        self.current_state = float(initial_state if initial_state is not None else open_val)

    def reset(self, initial_state: float = None):
        """Resets the internal latch state."""
        self.current_state = float(initial_state if initial_state is not None else self.open_val)

    def step(self, raw_gripper_action: Union[float, torch.Tensor]) -> float:
        """
        Processes a single predicted gripper output.

        Args:
            raw_gripper_action: Continuous prediction in [-1.0, 1.0].

        Returns:
            Discretized stable gripper state: +1.0 (Closed) or -1.0 (Open).
        """
        if isinstance(raw_gripper_action, torch.Tensor):
            val = raw_gripper_action.item()
        else:
            val = float(raw_gripper_action)

        # 1. Affine normalization: [-1, 1] -> [0, 1]
        g_bar = (val + 1.0) / 2.0

        # 2. Hysteresis latch
        if g_bar > self.high_thresh:
            self.current_state = self.close_val  # Close gripper (+1.0)
        elif g_bar < self.low_thresh:
            self.current_state = self.open_val   # Open gripper (-1.0)
        # else: retain previous state (hysteresis deadband prevents chattering)

        return self.current_state

    def process_chunk(self, raw_gripper_chunk: torch.Tensor) -> torch.Tensor:
        """
        Applies Schmitt trigger sequentially across a horizon chunk of gripper actions.

        Args:
            raw_gripper_chunk: Tensor of shape [H] or [1, H] in range [-1.0, 1.0]

        Returns:
            Discretized actions tensor of shape [H] with values in {0.0, 1.0}
        """
        flat = raw_gripper_chunk.view(-1)
        out = torch.zeros_like(flat)
        for t in range(flat.shape[0]):
            out[t] = self.step(flat[t])
        return out.view_as(raw_gripper_chunk)
