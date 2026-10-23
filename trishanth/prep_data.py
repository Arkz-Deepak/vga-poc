"""Replay LIBERO-Spatial demos -> per-task arrays (RGB, metric depth, Plücker rays, proprio, actions) + frozen SigLIP features.

data/proc/t{task}/  img.npy [N,2,224,224,3] u8 | depth.npy [N,2,28,28] f16 | ray.npy [N,2,7,7,6] f16
                    state.npy [N,11] | action.npy [N,7] | ep.npy [N] | feat.npy [N,2,49,768] f16
data/proc/lang.npy  [10,768] SigLIP text embeddings of the task instructions
"""
import glob, os, sys
import h5py, numpy as np, torch
from common import ROOT, bddl_path, frame, make_env, suite

POOL = int(os.environ.get("POOL", 20))  # demos per task kept as the few-shot sampling pool
OUT = os.path.join(ROOT, "data", "proc")
SIGLIP = "google/siglip-base-patch16-224"


def is_noop(a, prev):
    return np.linalg.norm(a[:6]) < 1e-4 and (prev is None or a[6] == prev[6])


def replay(task_id):
    t = suite().get_task(task_id)
    path = glob.glob(os.path.join(ROOT, "data", "raw", "libero_spatial", f"{t.name}_demo.hdf5"))[0]
    env = make_env(bddl_path(task_id), depth=True)
    env.reset()
    buf = {k: [] for k in ("img", "depth", "ray", "state", "action", "ep")}
    err = []
    with h5py.File(path, "r") as f:
        for e in range(POOL):
            d = f[f"data/demo_{e}"]
            states, actions, ee = d["states"][:], d["actions"][:], d["obs/ee_pos"][:]
            prev = None
            for i in range(len(actions)):
                if is_noop(actions[i], prev):
                    continue
                prev = actions[i]
                obs = env.set_init_state(states[i])
                fr = frame(obs, env.env.sim, depth=True)
                err.append(np.linalg.norm(fr["state"][:3] - ee[i]))
                buf["img"].append(fr["img"])
                buf["depth"].append(fr["depth"].reshape(2, 28, 8, 28, 8).mean((2, 4)))
                buf["ray"].append(fr["ray"])
                buf["state"].append(fr["state"])
                buf["action"].append(actions[i])
                buf["ep"].append(e)
    env.close()
    od = os.path.join(OUT, f"t{task_id}")
    os.makedirs(od, exist_ok=True)
    for k, v in buf.items():
        v = np.stack(v)
        np.save(os.path.join(od, f"{k}.npy"), v.astype(np.float16) if k in ("depth", "ray") else v)
    print(f"task {task_id}: {len(buf['ep'])} frames, replay eef err mean {np.mean(err)*1000:.2f} mm max {np.max(err)*1000:.2f} mm", flush=True)


@torch.no_grad()
def features():
    from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionModel
    vis = SiglipVisionModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).cuda().eval()
    for task_id in range(10):
        od = os.path.join(OUT, f"t{task_id}")
        img = np.load(os.path.join(od, "img.npy"), mmap_mode="r")
        feats = []
        for s in range(0, len(img), 64):
            x = torch.from_numpy(np.ascontiguousarray(img[s:s + 64])).cuda().flatten(0, 1).permute(0, 3, 1, 2)
            x = (x.half() / 255 - 0.5) / 0.5
            h = vis(pixel_values=x).last_hidden_state  # [B*2, 196, 768]
            h = h.view(-1, 14, 14, 768).permute(0, 3, 1, 2)
            h = torch.nn.functional.avg_pool2d(h, 2).flatten(2).transpose(1, 2)  # [B*2, 49, 768]
            feats.append(h.view(-1, 2, 49, 768).cpu().numpy())
        np.save(os.path.join(od, "feat.npy"), np.concatenate(feats).astype(np.float16))
        print(f"task {task_id}: features done", flush=True)
    tok = AutoTokenizer.from_pretrained(SIGLIP)
    txt = SiglipTextModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).cuda().eval()
    langs = [suite().get_task(i).language for i in range(10)]
    ids = tok(langs, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
    np.save(os.path.join(OUT, "lang.npy"), txt(**ids).pooler_output.float().cpu().numpy())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("replay", "all"):
        for task_id in range(10):
            replay(task_id)
    if stage in ("features", "all"):
        features()
