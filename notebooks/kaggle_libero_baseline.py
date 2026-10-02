"""
Kaggle Notebook Script: Baseline Exploration & Data Ingestion (LeRobot & LIBERO).

HOW TO RUN ON KAGGLE:
1. Go to https://www.kaggle.com/code and click "New Notebook".
2. On the right-side Settings panel:
   - Accelerator: Select 'GPU T4 x 2' or 'GPU P100'.
   - Internet: Toggle ON (required to download Hugging Face packages and datasets).
3. Paste the cells below into your Kaggle notebook and run them!

PURPOSE OF THIS EXPERIMENT ("Existing Technology"):
- Ingest and inspect the LIBERO spatial dataset using Hugging Face's LeRobot library.
- Filter episodes for the 3 target benchmark tasks:
  1. Bowl onto plate
  2. Soup into basket
  3. Plate to stove
- Inspect camera frames (RGB 256x256), proprioceptive robot state, and action trajectories.
- Compute the empirical normalization vectors (mu, sigma) required by the policy normalizer.
"""

# ==============================================================================
# Cell 1: Install LeRobot and Dependencies on Kaggle
# ==============================================================================
# !pip install --quiet git+https://github.com/huggingface/lerobot.git
# !pip install --quiet zarr einops scipy torchvision transformers peft

# ==============================================================================
# Cell 2: Test LeRobot & PyTorch Environment
# ==============================================================================
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# Cell 3: Fetch LIBERO Dataset from Hugging Face via LeRobot
# ==============================================================================
import json
import numpy as np

# Download and inspect dataset
print("Loading LIBERO dataset metadata from Hugging Face...")
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# dataset = LeRobotDataset("lerobot/libero", split="train")
# print(f"Total episodes available: {dataset.num_episodes}")
# print(f"Total frames: {dataset.num_frames}")
# print(f"Features in dataset: {dataset.features.keys()}")

# ==============================================================================
# Cell 4: Filter the 3 Target Spatial Tasks for our VGA PoC
# ==============================================================================
TARGET_TASKS = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "push_the_plate_to_the_front_of_the_stove"
]

print(f"Selected 3 PoC Benchmark Tasks:\n" + "\n".join(f"- {t}" for t in TARGET_TASKS))

# ==============================================================================
# Cell 5: Compute Action Normalization Statistics (mu_d, sigma_d, sigma_pos_sq, sigma_rot_sq)
# ==============================================================================
def compute_mock_or_real_action_stats(actions_np):
    """
    Computes mean and standard deviation for the 7D actions:
    [delta_x, delta_y, delta_z, r_x, r_y, r_z, gripper]
    """
    mu_d = np.mean(actions_np, axis=0).tolist()
    sigma_d = (np.std(actions_np, axis=0) + 1e-6).tolist()

    # Position variance (mean across dims 0, 1, 2)
    sigma_pos_sq = float(np.mean(np.array(sigma_d[:3]) ** 2))
    # Rotation variance (mean across dims 3, 4, 5)
    sigma_rot_sq = float(np.mean(np.array(sigma_d[3:6]) ** 2))

    stats = {
        "mu_d": mu_d,
        "sigma_d": sigma_d,
        "sigma_pos_sq": sigma_pos_sq,
        "sigma_rot_sq": sigma_rot_sq
    }
    return stats

# Example synthetic check:
dummy_actions = np.random.randn(1000, 7) * 0.05
dummy_actions[:, 6] = np.random.choice([-1.0, 1.0], size=1000)
stats = compute_mock_or_real_action_stats(dummy_actions)
print("\nComputed Action Statistics Example:")
print(json.dumps(stats, indent=2))
