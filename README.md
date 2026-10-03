# VGA: Vision-Geometry-Action Robotic Manipulation Policy

[![Model Parameters](https://img.shields.io/badge/Parameters-298.3M_%28%E2%89%A40.5B%20Ceiling%29-blue.svg)](https://github.com/Arkz-Deepak/vga-poc)
[![Inference Latency](https://img.shields.io/badge/Per--Step%20Latency-4.64ms_%28%E2%89%A418ms%20Target%29-brightgreen.svg)](https://github.com/Arkz-Deepak/vga-poc)
[![Control Loop](https://img.shields.io/badge/Control%20Frequency-154.7_--_215.5_Hz-orange.svg)](https://github.com/Arkz-Deepak/vga-poc)
[![Jerk Reduction](https://img.shields.io/badge/Kinematic%20Smoothness-89.1%25%20Jerk%20Reduction-blueviolet.svg)](https://github.com/Arkz-Deepak/vga-poc)

Lightweight ($\le 0.5\text{B}$ parameter) Proof of Concept (PoC) for the **Vision-Geometry-Action (VGA)** embodied AI policy, benchmarked on 3 spatial manipulation tasks from the **LIBERO-Spatial** benchmark using Hugging Face's **LeRobot** framework.

---

## 📊 Empirical Benchmark Results (Tesla T4 GPU)

The table below summarizes empirical measurements collected on **Tesla T4 GPUs** comparing the baseline `lerobot/smolvla_base` (450M parameters) against our **VGA Policy (298.3M parameters)**:

| Metric | SmolVLA-450M Baseline | 5-Shot VGA (1x T4) | 10-Shot VGA (2x T4 DDP) | Specification Target |
| :--- | :--- | :--- | :--- | :--- |
| **Total Parameter Budget** | 450.0M | **298.3M** *(34% lighter)* | **298.3M** *(34% lighter)* | $\le 500\text{M}$ |
| **16-Step Chunk Latency** | 643.75 ms | **74.31 ms** *(8.7× faster)* | **103.42 ms** *(6.2× faster)* | Real-Time |
| **Per-Step Motor Latency** | 40.23 ms | **4.64 ms** | **6.46 ms** | $\le \mathbf{18.0\text{ ms}}$ |
| **Real-Time Control Loop** | 24.8 Hz | **215.5 Hz** | **154.7 Hz** | $\ge \mathbf{50.0\text{ Hz}}$ |
| **Meets 50 Hz Hard Real-Time**| ❌ **FAILS** (40 ms) | ✅ **PASSES** (4.6 ms) | ✅ **PASSES** (6.5 ms) | $\le 18\text{ ms}$ |
| **Kinematic Jerk Metric** | 192.25 | 44.97 | **20.91** | Minimized |
| **Jerk Reduction (%)** | Baseline (0%) | **+76.6%** | **+89.1%** | $\ge \mathbf{30\%}$ |
| **Gripper Chatter Resistance** | No Hysteresis | Schmitt Trigger | Schmitt Trigger | Zero chattering |

---

## 🎯 Architecture & Innovations

### 1. Unified Space-to-Depth Projector ($9\times$ Visual Token Compression)
Standard VLAs feed hundreds of visual tokens directly into large transformer backbones, causing quadratic attention bottlenecks ($O(N^2)$). VGA deploys an invertible pixel-unshuffle operation followed by a linear projection:
$$\mathbf{X}_{vis} \in \mathbb{R}^{B \times 576 \times 768} \xrightarrow{\text{Space-to-Depth}} \mathbf{X}_{proj} \in \mathbb{R}^{B \times 64 \times 960}$$
- Compresses 576 patch tokens into just **64 tokens** ($9\times$ reduction).
- Preserves full high-frequency edge and boundary details without information loss.

### 2. CentroidRayRoPE: Zero-Overhead 3D Geometric Grounding
To bridge 2D image pixels and physical 3D robot workspace coordinates, VGA calculates unit 3D viewing rays $\mathbf{r}_{u, v} \in \mathbb{S}^2$ for every patch centroid using pinhole camera intrinsics:
$$\mathbf{r}_{u, v} = \frac{\mathbf{K}^{-1} [u, v, 1]^T}{\|\mathbf{K}^{-1} [u, v, 1]^T\|_2}$$
These unit vectors, along with the camera optical center in robot base coordinates $\mathbf{t}_{base} \in \mathbb{R}^3$, are injected via multi-head self-attention before token fusion, giving the model true physical spatial awareness with **zero extra sequence length**.

### 3. DiT Action Expert & 4-Step Euler Flow Matching
- **Action Chunking**: Predicts 16 future end-effector actions:
  $$\mathbf{A}_{t:t+16} = [\Delta x, \Delta y, \Delta z, r_x, r_y, r_z, \text{gripper}] \in \mathbb{R}^{16 \times 7}$$
- **Continuous Flow Matching**: Trained via optimal-transport probability paths:
  $$\mathbf{x}_t = (1 - (1 - \sigma_{min})t)\mathbf{x}_0 + t \mathbf{x}_1, \quad \mathbf{u}_t = \mathbf{x}_1 - (1 - \sigma_{min})\mathbf{x}_0$$
- **4-Step ODE Integration**: Generates the complete 16-step trajectory with only **4 function evaluations (NFE=4)** on a compact 33M DiT expert, achieving **4.64 ms per motor step**.

### 4. Taylor-Guarded Lie Algebra & Kinematic Smoothing
- **Singularity-Free Axis-Angle**: Standard matrix logarithm $\text{Log}(R)$ has numerical division-by-zero singularities as rotation angle $\theta \to 0$. VGA uses a 4th-order Taylor series expansion when $\theta < 10^{-4}$:
  $$\frac{\theta}{2 \sin \theta} = \frac{1}{2} + \frac{\theta^2}{12} + \frac{7\theta^4}{720} + \mathcal{O}(\theta^6)$$
  This guarantees strictly zero `NaN` values and stable gradient backpropagation.
- **Acceleration & Jerk Penalty**:
  $$\mathcal{L}_{kin} = \|\Delta^2 \mathbf{a}_t\|_2^2 + \beta_{jerk} \|\Delta^3 \mathbf{a}_t\|_2^2$$
  Reduced physical motor jerk by **89.1%** during fine-tuning.

### 5. Schmitt Trigger Gripper Controller
To eliminate erratic gripper chatter around contact thresholds, VGA applies an affine-scaled Schmitt trigger with a $[0.35, 0.65]$ deadband:
$$g_t = \begin{cases} 1 & \text{if } \hat{g}_t \ge 0.65 \\ 0 & \text{if } \hat{g}_t \le 0.35 \\ g_{t-1} & \text{otherwise (hysteresis)} \end{cases}$$

---

## 📁 Repository Structure

```text
vga-poc/
├── configs/
│   ├── poc_config.py          # Central hyperparameters (D_vis, D_lm, H=16, dt=0.02)
│   └── action_stats.json      # Dataset empirical normalization mean & std
├── controllers/
│   └── schmitt_trigger.py     # Affine scaling & hysteresis deadband controller
├── data/
│   └── dataset.py             # LeRobot v3.0 parquet / v2.0 loader with test splitting
├── losses/
│   └── kinematics.py          # Taylor-guarded Lie algebra & finite-difference jerk loss
├── models/
│   ├── backbones.py           # SigLIP-B/16 + Space-to-Depth + Ray-RoPE + SmolLM2
│   ├── dit_expert.py          # 12-layer Diffusion Transformer Action Expert (33M)
│   ├── projector.py           # UnifiedSpaceToDepthProjector (9x token compression)
│   ├── ray_rope.py            # CentroidRayRoPE 3D viewing ray attention pre-pass
│   └── vga_policy.py          # Unified VGAPolicy (298.3M params, predict_chunk)
├── results/                   # Evaluation plots, benchmark tables, JSON metrics, MP4 videos
├── DEVELOPMENT_TURNS.md       # Turn-by-turn engineering log and simulation post-mortem
├── POC_QNA_CHEATSHEET.md      # Team Q&A, elevator pitch, and interview cheat sheet
└── scripts/
    ├── baseline_kaggle_run.py # Dataset normalization & SmolVLA 450M latency benchmark
    ├── train_vga.py           # Single-GPU & Multi-GPU (2x T4 DDP) training pipeline
    ├── evaluate_libero.py     # Trajectory tracking, jerk reduction & latency benchmark
    ├── eval_mujoco_closed_loop.py # Closed-loop physics rollout simulation in MuJoCo
    ├── verify_phase1.py       # Phase 1 mathematical & geometric unit tests
    └── verify_phase2_vga.py   # Phase 4 end-to-end model & gradient flow tests
```

---

## 🛠️ Usage Instructions

### 1. Local Sanity Checks (CPU or Local GPU)
```bash
# Verify mathematical primitives (Ray-RoPE, Taylor guard, Kinematics, Schmitt Trigger)
python scripts/verify_phase1.py

# Verify end-to-end VGA policy architecture, autograd flow, and action queue
python scripts/verify_phase2_vga.py
```

### 2. Kaggle 5-Shot Demonstration Training (1x GPU T4)
```bash
!cd /kaggle/working/vga-poc && git pull origin main
!python /kaggle/working/vga-poc/scripts/train_vga.py --shots 5 --steps 300 --batch_size 8
```

### 3. Kaggle 10-Shot Demonstration Training (2x GPU T4, Distributed Data Parallel)
```bash
!cd /kaggle/working/vga-poc && git pull origin main
!torchrun --nproc_per_node=2 /kaggle/working/vga-poc/scripts/train_vga.py --shots 10 --steps 500 --batch_size 8
```

### 4. Offline Trajectory & Jerk Benchmark on Held-Out Test Data
```bash
!python /kaggle/working/vga-poc/scripts/evaluate_libero.py \
    --ckpt_5shot checkpoints/vga_libero_5shot.pt \
    --ckpt_10shot checkpoints/vga_libero_10shot.pt \
    --test_shots 5 \
    --skip_shots 10
```
This automatically computes:
1. Position Tracking MAE/RMSE (mm).
2. Rotation Axis-Angle Error (degrees).
3. Gripper Accuracy (%).
4. Empirical Jerk Reduction vs unconstrained baseline.
5. Real-Time Hardware Latency and Control Frequency.
6. Exports a 4-panel publication-ready comparison figure to `results/vga_benchmark_report.png`.

### 5. Closed-Loop MuJoCo Simulation Rollout (LIBERO-Spatial)
```bash
!python /kaggle/working/vga-poc/scripts/eval_mujoco_closed_loop.py \
    --checkpoint checkpoints/vga_libero_10shot.pt \
    --episodes 5
```
- **Live Physics**: Evaluates the model interacting step-by-step with MuJoCo contact physics.
- **Hysteresis Gripper Control**: Features affine Schmitt Trigger filtering with 60-step anti-slip hold and post-transport release latch.
- **MP4 Video Output**: Records full visual rollouts to `results/videos/`.

---

## 📖 Additional Documentation

- **[DEVELOPMENT_TURNS.md](DEVELOPMENT_TURNS.md)**: Comprehensive, turn-by-turn engineering chronology documenting every bug diagnosis, mathematical design decision, and simulation iteration (including the gripper chatter fix, approach guard, and friction calibration).
- **[POC_QNA_CHEATSHEET.md](POC_QNA_CHEATSHEET.md)**: 30-second elevator pitch, master metric comparison table, and quick-reference answers for technical reviews.

