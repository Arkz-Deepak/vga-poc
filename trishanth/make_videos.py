"""Side-by-side rollout videos: one MP4 per (task, start state), one tile per variant.

python make_videos.py videos/raw videos   # expects videos/raw/<variant>/t{task}_i{init}.npz from eval.py --record
"""
import glob, os, sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS = {"base": "Baseline", "depth": "+ depth supervision", "rays": "+ camera rays",
          "smooth": "+ smooth chunk joins", "full": "All three add-ons",
          "base+ema": "Baseline + moving-avg filter", "smolvla": "SmolVLA-450M"}
TILE, INSET, BAR, TITLE, FPS, HOLD = 336, 96, 30, 60, 20, 30


def font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_LABEL, F_TITLE, F_NOTE, F_BANNER = font(17), font(19), font(14), font(22)


def tile(views, label, outcome=None):
    """views [2,224,224,3] (front, wrist) -> labelled tile; outcome=(success, steps) once the episode is over."""
    img = Image.new("RGB", (TILE, TILE + BAR), (24, 26, 25))
    img.paste(Image.fromarray(views[0]).resize((TILE, TILE)), (0, BAR))
    wrist = Image.fromarray(views[1]).resize((INSET, INSET))
    img.paste((235, 235, 235), (TILE - INSET - 8, TILE + BAR - INSET - 8, TILE - 4, TILE + BAR - 4))
    img.paste(wrist, (TILE - INSET - 6, TILE + BAR - INSET - 6))
    d = ImageDraw.Draw(img)
    d.text((10, 6), label, font=F_LABEL, fill=(235, 235, 235))
    if outcome:
        ok, steps = outcome
        text = f"SUCCESS  step {steps}" if ok else f"FAILED  ({steps} steps)"
        d.rectangle((0, BAR + TILE // 2 - 22, TILE, BAR + TILE // 2 + 22), fill=(28, 120, 70) if ok else (170, 45, 40))
        w = d.textlength(text, font=F_BANNER)
        d.text(((TILE - w) / 2, BAR + TILE // 2 - 13), text, font=F_BANNER, fill=(255, 255, 255))
    return np.asarray(img)


def compose(raw_dir, out_dir):
    from common import suite
    variants = [v for v in LABELS if os.path.isdir(os.path.join(raw_dir, v))]
    episodes = sorted({os.path.basename(f) for v in variants for f in glob.glob(os.path.join(raw_dir, v, "t*_i*.npz"))})
    cols = 3 if len(variants) <= 6 else 4
    rows = -(-len(variants) // cols)
    os.makedirs(out_dir, exist_ok=True)
    for ep in episodes:
        runs = {v: np.load(os.path.join(raw_dir, v, ep)) for v in variants if os.path.exists(os.path.join(raw_dir, v, ep))}
        T = max(len(r["video"]) for r in runs.values()) + HOLD
        task = int(ep[1:ep.index("_")])
        title = suite().get_task(task).language
        note = f"LIBERO-Spatial task {task}, start state {ep[ep.index('_i') + 2:-4]}  ·  5 demos/task, seed 0  ·  real time (20 Hz)  ·  inset: wrist camera"
        W, Hh = cols * TILE, TITLE + rows * (TILE + BAR)
        path = os.path.join(out_dir, ep.replace(".npz", ".mp4"))
        with imageio.get_writer(path, fps=FPS, codec="libx264", quality=7, macro_block_size=16) as w:
            for t in range(T):
                canvas = Image.new("RGB", (W, Hh + (-Hh) % 16), (14, 16, 15))
                d = ImageDraw.Draw(canvas)
                d.text((12, 9), title, font=F_TITLE, fill=(240, 240, 240))
                d.text((12, 35), note, font=F_NOTE, fill=(165, 172, 168))
                for n, (v, r) in enumerate(runs.items()):
                    vid = r["video"]
                    over = t >= len(vid) - 1
                    tl = tile(vid[min(t, len(vid) - 1)], LABELS[v], (bool(r["success"]), len(vid) - 1) if over else None)
                    canvas.paste(Image.fromarray(tl), ((n % cols) * TILE, TITLE + (n // cols) * (TILE + BAR)))
                w.append_data(np.asarray(canvas))
        print("wrote", path, f"({T} frames, {os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    compose(sys.argv[1], sys.argv[2])
