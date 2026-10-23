"""Resumable ablation grid + paired statistics.

python experiments.py run [--shots 5 10] [--seeds 0 1 2]   # trains/evaluates whatever is missing
python experiments.py summary                               # -> results/summary.md
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("RUNS_DIR", os.path.join(ROOT, "runs"))  # results/runs in the shared repo
PY = sys.executable
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet downloads hang on this machine
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # all models are cached; a network outage once killed a SmolVLA eval
ARMS = {"base": [], "depth": ["--depth"], "rays": ["--rays"], "smooth": ["--smooth"], "full": ["--depth", "--rays", "--smooth"]}
EPISODES = 20  # per task -> 200 rollouts per run
WORKERS = 2    # each sim worker ~2 GB RSS + 0.7 GB GPU-shared RAM; 3+ page-thrash this 16 GB laptop


def sh(cmd, tries=3):
    for i in range(tries):  # sim workers occasionally die at start-up on Windows; runs are idempotent
        print(">>", " ".join(cmd), flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode == 0:
            return
    raise RuntimeError(f"failed {tries}x: {cmd}")


def run(shots_list, seeds, smol_seeds, smol_steps, smol_accum=1, smol_name="smolvla", smol_script="smolvla_baseline.py", smol_batch=16):
    if sys.platform == "win32":  # keep the PC from idle-sleeping while this process runs (released on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    for shots in shots_list:
        for seed in seeds:
            for arm, flags in ARMS.items():
                out = os.path.join("runs", f"{arm}_s{shots}_seed{seed}")
                if not os.path.exists(os.path.join(ROOT, out, "ckpt.pt")):
                    sh([PY, "vga.py", "--shots", str(shots), "--seed", str(seed), "--out", out, *flags])
                if not os.path.exists(os.path.join(ROOT, out, "eval.json")):
                    sh([PY, "eval.py", os.path.join(out, "ckpt.pt"), "--episodes", str(EPISODES), "--workers", str(WORKERS)])
                if arm == "base" and seed == 0 and not os.path.exists(os.path.join(ROOT, out, "eval_ema0.5.json")):
                    sh([PY, "eval.py", os.path.join(out, "ckpt.pt"), "--episodes", str(EPISODES), "--workers", str(WORKERS), "--ema", "0.5"])
        for seed in smol_seeds:  # external baseline on the identical split (~1.5 h per run on this laptop)
            out = os.path.join("runs", f"{smol_name}_s{shots}_seed{seed}")
            if not os.path.exists(os.path.join(ROOT, out, "stats.npz")):
                sh([PY, smol_script, "--shots", str(shots), "--seed", str(seed), "--steps", str(smol_steps),
                    "--batch", str(smol_batch), "--accum", str(smol_accum), "--out", out])
            if not os.path.exists(os.path.join(ROOT, out, "eval.json")):
                sh([PY, "eval.py", os.path.join(out, "model"), "--episodes", str(EPISODES), "--workers", str(WORKERS)])


def load(pattern):
    return {os.path.basename(os.path.dirname(f)): json.load(open(f)) for f in glob.glob(os.path.join(RUNS, pattern))}


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def summary():
    ev, ema = load("*_s*_seed*/eval.json"), load("base_s*_seed*/eval_ema0.5.json")
    ev.update({k.replace("base_", "base+ema_"): v for k, v in ema.items()})
    smol = load("smolvla_s*_seed*/eval.json")
    ev.update(smol)
    lines = ["| arm | shots | seeds | success (95% CI) | Δ vs base, paired (95% CI) | EE jerk RMS m/s³ | seam/inner jump | latency ms |",
             "|---|---|---|---|---|---|---|---|"]
    arms = list(ARMS) + ["base+ema", "smolvla", "smolvla_eq", "xvla_eq"]  # smolvla_eq: same budget as the compact model
    for shots in (5, 10):
        for arm in arms:
            runs = {k: v for k, v in ev.items() if k.startswith(f"{arm}_s{shots}_seed")}
            if not runs:
                continue
            succ, diffs, jerk, seam, lat = [], [], [], [], []
            for name, r in runs.items():
                eps = {(e["task"], e["init"]): e for e in r["per_episode"]}
                succ += [e["success"] for e in eps.values()]
                jerk.append(r["ee_jerk_rms"])
                seam.append(r["seam_jump"] / max(r["inner_jump"], 1e-9))
                lat.append(r["latency_ms"])
                base = ev.get(name.replace(arm, "base", 1))
                if arm != "base" and base:
                    beps = {(e["task"], e["init"]): e for e in base["per_episode"]}
                    diffs += [eps[k]["success"] - beps[k]["success"] for k in eps if k in beps]
            m, h = ci(succ)
            d = "" if not diffs else "{:+.1f} ± {:.1f}".format(*(100 * v for v in ci(diffs)))
            lines.append(f"| {arm} | {shots} | {len(runs)} | {100*m:.1f} ± {100*h:.1f} | {d} | {np.mean(jerk):.2f} | {np.mean(seam):.2f} | {np.mean(lat):.0f} |")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    md = "\n".join(lines)
    open(os.path.join(ROOT, "results", "summary.md"), "w", encoding="utf-8").write(md + "\n")
    print(md)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "summary"])
    p.add_argument("--shots", type=int, nargs="+", default=[5, 10])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--smol-seeds", type=int, nargs="*", default=[0])
    p.add_argument("--smol-steps", type=int, default=3000)
    p.add_argument("--smol-accum", type=int, default=1)
    p.add_argument("--smol-name", default="smolvla")
    p.add_argument("--smol-script", default="smolvla_baseline.py")  # or xvla_baseline.py
    p.add_argument("--smol-batch", type=int, default=16)
    a = p.parse_args()
    run(a.shots, a.seeds, a.smol_seeds, a.smol_steps, a.smol_accum, a.smol_name, a.smol_script, a.smol_batch) if a.cmd == "run" else summary()
