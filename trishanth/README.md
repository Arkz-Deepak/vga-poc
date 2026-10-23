# VGA ideas: controlled few-shot ablation on LIBERO-Spatial

*Trishanth Mellimi · Oct 2026 · run locally on one laptop (RTX 4060 Laptop, 8 GB)*

This folder tests the three VGA ideas as **on/off add-ons to one fixed compact policy**, so each idea's effect can be measured on its own:

- **Depth supervision during training (`--depth`).** Quantized metric depth is predicted from mid-network visual tokens. The head is removed at test time.
- **Camera rays (`--rays`).** A per-token Plücker ray (direction plus moment, world frame) is added to visual tokens. Wrist-camera rays move with the arm.
- **Smooth chunk joins (`--smooth`).** The action head is conditioned on the last 4 executed actions and penalized for acceleration and jerk across chunk boundaries.

The study also includes two outside baselines, **SmolVLA-450M** and **X-VLA-0.9B**, fine-tuned on the same demos with the same training budget.

It complements the main VGA-PoC model in the repo root. The two use different evaluation protocols, so their numbers aren't directly comparable (see [Relation to the main VGA-PoC](#relation-to-the-main-vga-poc)).

Team summary page: https://claude.ai/artifact/KpGkSmw5AUAY7DQAWL31as

---

## Findings

1. **Smooth chunk joins are the one add-on with a clear, repeatable benefit.** Gripper jerk drops about 35% at both demo budgets. The jump at chunk boundaries falls from about 2.8× a normal step to 1.05×. The cost in success (−4.2 and −2.5 points) is within noise.
2. **Depth supervision and camera rays have no reliable effect on success.** Depth was +2.8 ± 5.1 with 5 demos and −1.5 ± 5.1 with 10. Rays were +0.3 and −0.2. LIBERO's fixed cameras give rays little to add, so they should be tested under camera-viewpoint shifts.
3. **A training-free moving-average action filter is not a substitute.** It cuts jerk only 22–24% and costs 9.5–11 points of success.
4. **Pretrained VLAs at equal training budget:**
   - With 5 demos, everything ties at 54–60%.
   - With 10 demos, X-VLA-0.9B is best at 71.0%: +9.5 ± 9.4 over the compact baseline (just significant at one seed), and +2.5 over the compact model with all three add-ons.
   - X-VLA is also the jerkiest model, with chunk-boundary jumps of 3.4×. Adding smooth chunk joins to X-VLA is the most promising follow-up.
5. **Single-seed comparisons on LIBERO mislead.** Three early "results" reversed with more seeds or a fair training budget:
   - camera rays went from −16.5 to +0.3 points,
   - the combined model went from +7 to −2,
   - SmolVLA's deficit went from 11–13 points to about 0.

   In single runs, 83 of 200 paired rollouts disagree. Use at least 3 seeds and paired tests.

## Results

Each cell is 200 rollouts: all 10 LIBERO-Spatial tasks × the same 20 fixed starting states, with a 220-step limit. Rows with 3 seeds pool 600 rollouts. Δ is a paired difference against the baseline on the same starting states, with a 95% confidence interval. The filter, SmolVLA and X-VLA rows are seed 0 only and are compared with the seed-0 baseline. Jerk is the RMS third derivative of gripper position at 20 Hz. Latency is one batch-1 plan on the laptop GPU.

**5 demos per task**

| Variant | Seeds | Success | Δ vs baseline | Jerk (m/s³) | Chunk-boundary jump ÷ normal step | Latency |
|---|---|---|---|---|---|---|
| Compact baseline (7.9M trainable) | 3 | 55.2 ± 4.0% | — | 7.64 | 2.88× | 18 ms |
| + depth supervision | 3 | 58.0 ± 4.0% | +2.8 ± 5.1 | 8.06 | 3.01× | 18 ms |
| + camera rays | 3 | 55.5 ± 4.0% | +0.3 ± 4.9 | 7.54 | 2.88× | 19 ms |
| + smooth chunk joins | 3 | 51.0 ± 4.0% | −4.2 ± 5.1 | **4.82** | **1.06×** | 20 ms |
| All three | 3 | 54.3 ± 4.0% | −0.8 ± 5.2 | 5.04 | 1.05× | 26 ms |
| Baseline + moving-average filter | 1 | 50.5 ± 6.9% | −9.5 ± 7.7 | 5.22 | 1.54× | same model |
| SmolVLA-450M, equal budget | 1 | 54.5 ± 6.9% | −5.5 ± 8.7 | 7.32 | 2.20× | ~1,050 ms |
| X-VLA-0.9B, equal budget | 1 | 54.0 ± 6.9% | −6.0 ± 9.2 | 7.83 | 2.86× | ~195 ms |

**10 demos per task**

| Variant | Seeds | Success | Δ vs baseline | Jerk (m/s³) | Chunk-boundary jump ÷ normal step | Latency |
|---|---|---|---|---|---|---|
| Compact baseline | 3 | 65.0 ± 3.8% | — | 7.95 | 2.83× | 25 ms |
| + depth supervision | 3 | 63.5 ± 3.9% | −1.5 ± 5.1 | 8.31 | 2.97× | 22 ms |
| + camera rays | 3 | 64.8 ± 3.8% | −0.2 ± 4.9 | 7.90 | 2.80× | 22 ms |
| + smooth chunk joins | 3 | 62.5 ± 3.9% | −2.5 ± 5.1 | **5.05** | **1.05×** | 24 ms |
| All three | 3 | 63.0 ± 3.9% | −2.0 ± 4.8 | 5.25 | 1.04× | 29 ms |
| Baseline + moving-average filter | 1 | 50.5 ± 6.9% | −11.0 ± 8.3 | 6.43 | 1.69× | same model |
| SmolVLA-450M, equal budget | 1 | 63.0 ± 6.7% | +1.5 ± 9.2 | 7.60 | 2.15× | ~380 ms |
| **X-VLA-0.9B, equal budget** | 1 | **71.0 ± 6.3%** | +9.5 ± 9.4 | 8.90 | 3.42× | ~180 ms |

The full table, including the earlier short-budget SmolVLA runs (48k samples, which underperformed because they were undertrained), is in [`results/summary.md`](results/summary.md). Raw per-episode results are in [`results/runs/`](results/runs).

**Rollout videos** (seed-0 models, 5 demos/task, six variants side by side, real time, wrist-camera inset):
[task 0](media/videos/t0_i0.mp4) · [task 4 (drawer)](media/videos/t4_i0.mp4) · [task 6](media/videos/t6_i0.mp4) · [task 9](media/videos/t9_i0.mp4)

## Setup used

| | |
|---|---|
| Benchmark | LIBERO-Spatial, all 10 tasks, one language-conditioned multi-task policy |
| Demos | Each seed draws 5 or 10 demos per task from the first 20 human demos. Demos are re-rendered from saved simulator states at 224×224 (front and wrist cameras) and no-op frames are dropped. |
| Compact policy | Frozen SigLIP-B/16 features for both cameras (2×2-pooled to 49 tokens each) and a frozen SigLIP text embedding, followed by a 4-layer transformer (d = 256) and a 4-layer flow-matching action head. Predicts 16-step chunks of 7-D relative actions, executes 8, uses 10 Euler steps. 7.9M trainable parameters. |
| Training budget | Every model gets 8,000 optimizer steps × batch 64 = 512k samples, with AdamW, warmup and cosine decay. SmolVLA and X-VLA use gradient accumulation to fit in 8 GB. |
| SmolVLA | `lerobot/smolvla_base`, vision-language backbone frozen, action expert trained (100M params), z-scored relative actions |
| X-VLA | `lerobot/xvla-base`, Florence-2 backbone frozen, policy transformer and soft prompts trained (311M params), bf16, `auto` action mode with the same z-scored relative actions |
| Evaluation | 20 fixed LIBERO starting states per task, 10 settle steps, at most 220 steps, success read from the simulator. Rollouts run in parallel simulator processes with batched GPU inference. |

## Relation to the main VGA-PoC

The repo-root model (v11) reports 73% on 10 trials per task. This study uses a different protocol: 20 fixed starting states per task, re-rendered 224×224 images, 3 training seeds and a fixed 512k-sample budget. A natural next step is to evaluate v11 with `eval.py` on the same 200 starting states, so both lines of work share one scoreboard.

## Files

| File | What it does |
|---|---|
| `common.py` | LIBERO helpers: environment creation, upright images, metric depth, per-token Plücker rays, 11-D state (position, 6D rotation, gripper), torch-free simulator worker |
| `prep_data.py` | Replays demos into `data/proc/` (images, depth, rays, state, actions) and caches SigLIP image and text features |
| `vga.py` | Compact policy and training script (`--depth`, `--rays`, `--smooth`) |
| `eval.py` | Closed-loop evaluation: parallel simulator workers, CUDA-graphed sampler, success, jerk and chunk-boundary metrics, `--record` for videos |
| `smolvla_baseline.py`, `xvla_baseline.py` | Fine-tune SmolVLA and X-VLA on the identical split; shared data loading and training loop |
| `experiments.py` | Resumable grid runner and paired-statistics summary (`run`, `summary`) |
| `make_videos.py` | Side-by-side comparison videos from `eval.py --record` output |
| `requirements-lock.txt` | Exact package versions used (Python 3.10, PyTorch 2.7.1 with CUDA 12.6) |

## Reproducing

```bash
# Environment: Python 3.10 venv, then
uv pip install -r requirements-lock.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO third_party/LIBERO   # put third_party/LIBERO on sys.path (.pth file)
# LIBERO demos: huggingface dataset yifengzhu-hf/LIBERO-datasets (libero_spatial/*) -> data/raw/

python prep_data.py                                   # ~15 min: replay demos and cache features
python experiments.py run --shots 5 10 --seeds 0 1 2  # compact grid, plus SmolVLA seed 0 (smolvla_baseline.py)
python experiments.py run --shots 5 10 --seeds 0 1 2 --smol-seeds 0 --smol-steps 8000 --smol-accum 4 --smol-name smolvla_eq
python experiments.py run --shots 5 10 --seeds 0 1 2 --smol-seeds 0 --smol-steps 8000 --smol-batch 8 --smol-accum 8 --smol-name xvla_eq --smol-script xvla_baseline.py
RUNS_DIR=results/runs python experiments.py summary   # regenerate the tables from the saved results
```

**Running LIBERO natively on Windows** needed five local fixes:

1. Copy `mujoco/mujoco.dll` into `robosuite/utils/`.
2. Create `robosuite/macros_private.py` with `MUJOCO_GPU_RENDERING = False`, because EGL only exists on Linux.
3. In LIBERO's `benchmark/__init__.py`, call `torch.load(..., weights_only=False)`.
4. In `robosuite/models/assets/base.xml`, replace the `<size nconmax=5000 njmax=5000>` tag with `<size memory="64M"/>`. MuJoCo 3 otherwise preallocates 1.3 GB per environment, and peak use measured here is 9.5 MB.
5. Write a project-local `.libero/config.yaml`, or set `LIBERO_CONFIG_PATH`, so LIBERO doesn't prompt interactively.

`experiments.py` sets `HF_HUB_OFFLINE=1`. Once models are cached, a network outage can't break evaluation; one did break an evaluation during this study.

**Runtime on the laptop:**
- each compact variant: about 7 minutes of training and about 30 minutes of evaluation (2 simulator workers, limited by RAM),
- each equal-budget SmolVLA run: about 7–9 hours,
- each equal-budget X-VLA run: about 10 hours.

## Caveats

- Simulation only, on one suite (LIBERO-Spatial) with fixed cameras.
- Image features come from a frozen encoder with no image augmentation. That's the same for every compact variant, but it caps absolute success at about 65% with 10 demos.
- SmolVLA and X-VLA have one seed per budget. Their pretraining covers other robots, so the prior transfers to LIBERO's Franka arm only partially.
- Checkpoints aren't committed because of size; they can be regenerated with the commands above.
