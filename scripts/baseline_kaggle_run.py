"""
VGA PoC: Baseline Exploration (LIBERO-Spatial & SmolVLA-450M)

This script performs end-to-end ingestion and baseline benchmarking:
1. Ingests the LIBERO-Spatial manipulation dataset via LeRobot.
2. Identifies target spatial tasks.
3. Computes empirical action normalization statistics (mu_d, sigma_d, sigma_pos_sq, sigma_rot_sq).
4. Loads baseline lerobot/smolvla_base (450M parameter VLA).
5. Runs GPU forward inference and benchmarks latency against our 18 ms target budget.
"""

import sys
import types
import pathlib
import time
import json
import torch
import numpy as np

# 1. Upstream compatibility guard for PyAV
try:
    import av
    if not hasattr(av, "option"):
        av.option = types.ModuleType("av.option")
        av.option.Option = object
except ImportError:
    pass

# 2. Namespace bypass for experimental lerobot policies
try:
    import lerobot
    policies_dir = pathlib.Path(lerobot.__file__).parent / "policies"
    if policies_dir.exists():
        dummy_policies = types.ModuleType("lerobot.policies")
        dummy_policies.__path__ = [str(policies_dir)]
        dummy_policies.__file__ = str(policies_dir / "__init__.py")
        sys.modules["lerobot.policies"] = dummy_policies
except ImportError:
    pass


def verify_accelerator():
    print("=" * 60)
    print("1. Accelerator & Environment Check")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM: {vram_gb:.2f} GB")
    return "cuda" if cuda_avail else "cpu"


def load_dataset(repo_id="lerobot/libero_spatial_image"):
    print("\n" + "=" * 60)
    print(f"2. Fetching Dataset: {repo_id}")
    print("=" * 60)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        dataset = LeRobotDataset(repo_id)
    except Exception as e:
        print(f"Notice: Failed to load {repo_id} ({e}), falling back to lerobot/libero_spatial...")
        repo_id = "lerobot/libero_spatial"
        dataset = LeRobotDataset(repo_id)

    print(f"✅ Loaded {repo_id}")
    print(f"- Total episodes: {dataset.num_episodes}")
    print(f"- Total frames: {dataset.num_frames}")
    print(f"- FPS: {getattr(dataset, 'fps', 10)}")
    return dataset


def compute_normalization_stats(dataset, sample_frames=5000, output_path="configs/action_stats.json"):
    print("\n" + "=" * 60)
    print(f"3. Computing Empirical Normalization Statistics ({sample_frames} frames)")
    print("=" * 60)

    n_samples = min(sample_frames, len(dataset))
    actions = [dataset[i]["action"].numpy() for i in range(n_samples)]
    actions_np = np.stack(actions, axis=0)

    mu_d = np.mean(actions_np, axis=0).tolist()
    sigma_d = (np.std(actions_np, axis=0) + 1e-6).tolist()
    sigma_pos_sq = float(np.mean(np.array(sigma_d[:3]) ** 2))
    sigma_rot_sq = float(np.mean(np.array(sigma_d[3:6]) ** 2))

    stats = {
        "mu_d": mu_d,
        "sigma_d": sigma_d,
        "sigma_pos_sq": sigma_pos_sq,
        "sigma_rot_sq": sigma_rot_sq,
    }

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Empirical Normalization Statistics:")
    print(json.dumps(stats, indent=2))
    print(f"✅ Saved to {output_path}")
    return stats


def test_baseline_policy(dataset, device="cuda"):
    print("\n" + "=" * 60)
    print("4. Testing SmolVLA-450M Baseline Policy")
    print("=" * 60)

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print("Loading weights for 'lerobot/smolvla_base'...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)
    policy.eval()

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"✅ SmolVLA loaded successfully on {device} ({total_params / 1e6:.1f}M params)")

    # Prepare batch from sample frame 0
    sample = dataset[0]
    task_desc = sample.get("task", "pick up the black bowl between the plate and the ramekin and place it on the plate")

    def prep_img(t):
        img = t.unsqueeze(0).to(device)
        return img.float() / 255.0 if img.dtype == torch.uint8 else img.float()

    batch = {
        "observation.images.camera1": prep_img(sample["observation.images.image"]),
        "observation.images.camera2": prep_img(sample["observation.images.wrist_image"]),
        "observation.images.camera3": prep_img(sample["observation.images.image"]),
        "observation.state": sample["observation.state"].unsqueeze(0).float().to(device),
    }

    # Language tokenization
    seq_len = 48
    tok = None
    try:
        tok = getattr(getattr(getattr(policy, "model", None), "vlm_with_expert", None), "processor", None)
        tok = getattr(tok, "tokenizer", None)
    except Exception:
        pass

    if tok is None:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        except Exception:
            pass

    if tok is not None:
        tokenized = tok(task_desc, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)
        batch["observation.language.tokens"] = tokenized["input_ids"].to(device)
        batch["observation.language.attention_mask"] = tokenized["attention_mask"].bool().to(device)
    else:
        batch["observation.language.tokens"] = torch.zeros((1, seq_len), dtype=torch.long, device=device)
        batch["observation.language.attention_mask"] = torch.ones((1, seq_len), dtype=torch.bool, device=device)

    batch["observation.language_tokens"] = batch["observation.language.tokens"]
    batch["observation.language_attention_mask"] = batch["observation.language.attention_mask"]

    # Action prediction
    policy.reset()
    with torch.no_grad():
        action = policy.select_action(batch)
        print("✅ Forward action prediction succeeded!")
        print(f"Action shape: {action.shape}, Values: {action.cpu().numpy().tolist()}")

    # Benchmark forward latency
    if device == "cuda":
        print("\nBenchmarking action chunk forward latency over 25 iterations...")
        for _ in range(5):
            policy.reset()
            _ = policy.select_action(batch)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(25):
            policy.reset()
            _ = policy.select_action(batch)
        torch.cuda.synchronize()
        latency_ms = (time.time() - t0) / 25.0 * 1000.0

        print(f"SmolVLA Action-Chunk Latency: {latency_ms:.2f} ms ({1000.0 / latency_ms:.1f} inferences/sec)")
        print("Target Latency Budget: <= 18.0 ms (VGA Target)")
        if latency_ms <= 18.0:
            print("🚀 Meets the <= 18 ms real-time robotics budget!")
        else:
            print(f"⚠️ Latency is {latency_ms:.1f} ms (> 18 ms budget) -> Motivates our 4-step DiT optimization!")


if __name__ == "__main__":
    dev = verify_accelerator()
    ds = load_dataset()
    compute_normalization_stats(ds)
    test_baseline_policy(ds, device=dev)
