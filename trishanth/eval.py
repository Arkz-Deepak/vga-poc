"""Closed-loop LIBERO-Spatial evaluation: parallel sim workers, batched GPU policy inference.

python eval.py runs/x/ckpt.pt --episodes 20 [--ema 0.5]  -> runs/x/eval[_ema0.5].json
"""
import argparse, json, os, subprocess, sys, time
from multiprocessing.connection import Listener
import numpy as np, torch, torch.nn.functional as F

MAX_STEPS, WAIT, EXEC, DT = 220, 10, 8, 0.05  # LIBERO-Spatial horizon, settle steps, executed chunk prefix, 20 Hz


class Policy:
    def __init__(self, ckpt):
        from transformers import SiglipVisionModel
        from vga import PROC, P, load_policy
        self.model, self.stats, self.args = load_policy(ckpt)
        self.vis = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=torch.float16).cuda().eval()
        self.lang = torch.from_numpy(np.load(os.path.join(PROC, "lang.npy"))).float().cuda()
        self.P, self.graphs = P, {}

    @torch.no_grad()
    def __call__(self, frames, tids, histories):
        B, s = len(frames), self.stats
        img = torch.from_numpy(np.stack([f["img"] for f in frames])).cuda().flatten(0, 1).permute(0, 3, 1, 2)
        h = self.vis(pixel_values=(img.half() / 255 - 0.5) / 0.5).last_hidden_state
        h = F.avg_pool2d(h.view(-1, 14, 14, 768).permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)
        feat = h.reshape(B, 2, 49, 768).float()
        ray = torch.from_numpy(np.stack([f["ray"] for f in frames])).cuda().flatten(2, 3)
        state = (torch.from_numpy(np.stack([f["state"] for f in frames])).cuda() - s["s_mean"]) / s["s_std"]
        prefix = torch.zeros(B, self.P, 7, device="cuda")
        pmask = torch.zeros(B, self.P, device="cuda")
        for b, hist in enumerate(histories):
            if hist:
                last = torch.tensor(np.array(hist[-self.P:]), device="cuda").float()
                prefix[b, -len(last):] = (last - s["a_mean"]) / s["a_std"]
                pmask[b, -len(last):] = 1
        g, st, out = self._graph(B)
        for k, v in (("feat", feat), ("ray", ray), ("lang", self.lang[torch.tensor(tids, device="cuda")]),
                     ("state", state), ("prefix", prefix), ("pmask", pmask)):
            st[k].copy_(v)
        st["x0"].normal_()
        g.replay()
        return (out * s["a_std"] + s["a_mean"]).cpu().numpy()

    def _graph(self, B):
        """CUDA-graph the trunk + 10-step sampler per batch size (Windows kernel-launch overhead dominates otherwise)."""
        if B not in self.graphs:
            z = lambda *shape: torch.zeros(*shape, device="cuda")
            st = dict(feat=z(B, 2, 49, 768), ray=z(B, 2, 49, 6), lang=z(B, 768), state=z(B, 11),
                      prefix=z(B, self.P, 7), pmask=z(B, self.P), x0=z(B, 16, 7))
            f = lambda: self.model.sample(self.model.encode(st["feat"], st["ray"], st["lang"], st["state"])[0],
                                          st["prefix"], st["pmask"], x=st["x0"].clone())
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    f()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = f()
            self.graphs[B] = (g, st, out)
        return self.graphs[B]


def metrics(ep):
    p, a = np.array(ep["eef"]), np.array(ep["acts"])
    jerk = np.diff(p, 3, axis=0) / DT ** 3
    da = np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1)
    seam = np.zeros(len(da), bool)
    seam[[b - 1 for b in ep["bounds"] if 0 < b <= len(da)]] = True
    return dict(
        task=ep["job"][0], init=ep["job"][1], success=ep["success"], steps=len(a),
        ee_jerk_rms=float(np.sqrt((jerk ** 2).sum(1).mean())) if len(jerk) else 0.0,
        seam_jump=float(da[seam].mean()) if seam.any() else 0.0,
        inner_jump=float(da[~seam].mean()) if (~seam).any() else 0.0,
    )


def evaluate(ckpt, episodes, n_workers, ema, jobs=None, record=None):
    if "smolvla" in ckpt:
        from smolvla_baseline import SmolVLAAgent
        pol = SmolVLAAgent(ckpt)
    elif "xvla" in ckpt:
        from xvla_baseline import XVLAAgent
        pol = XVLAAgent(ckpt)
    else:
        pol = Policy(ckpt)
    # latency of one batch-1 chunk inference (SigLIP x2 + trunk + 10 Euler steps)
    dummy = dict(img=np.zeros((2, 224, 224, 3), np.uint8), ray=np.zeros((2, 7, 7, 6), np.float32), state=np.zeros(11, np.float32))
    for _ in range(5):
        pol([dummy], [0], [[]])
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        pol([dummy], [0], [[]])
    torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 20 * 1000

    pending = {t: [j for j in jobs if j[0] == t] if jobs else [(t, k) for k in range(episodes)] for t in range(10)}
    wtask = [None] * n_workers

    def next_job(w):
        """Keep each worker on one task while it has episodes left (env re-creation costs seconds)."""
        if not pending.get(wtask[w]):
            left = [t for t in pending if pending[t]]
            if not left:
                return None
            idle = [t for t in left if t not in wtask]
            wtask[w] = (idle or left)[0]
        return pending[wtask[w]].pop(0)

    from common import ROOT, bddl_path, suite
    bddls = [bddl_path(t) for t in range(10)]
    inits = [suite().get_task_init_states(t) for t in range(10)]
    lst = Listener(("127.0.0.1", 0), authkey=b"libero")
    procs = [subprocess.Popen([sys.executable, os.path.join(ROOT, "common.py"), str(lst.address[1])], cwd=ROOT)
             for _ in range(n_workers)]
    pipes = [lst.accept() for _ in range(n_workers)]
    slots, done_eps = [None] * n_workers, []

    def reset(ws):
        sent = []
        for w in ws:
            job = next_job(w)
            slots[w] = job and dict(job=job, queue=[], acts=[], eef=[], bounds=[], success=False)
            if job:
                pipes[w].send(("reset", bddls[job[0]], inits[job[0]][job[1]], WAIT))
                sent.append(w)
        for w in sent:
            slots[w]["frame"], eef, _ = pipes[w].recv()
            slots[w]["eef"].append(eef)
            slots[w]["video"] = [slots[w]["frame"]["img"]] if record else None

    reset(range(n_workers))
    t0, tm = time.time(), dict(policy=0.0, step=0.0, reset=0.0)
    while any(slots):
        live = [w for w in range(n_workers) if slots[w]]
        need = [w for w in live if not slots[w]["queue"]]
        tp = time.time()
        if need:
            chunks = pol([slots[w]["frame"] for w in need], [slots[w]["job"][0] for w in need], [slots[w]["acts"] for w in need])
            for w, c in zip(need, chunks):
                slots[w]["queue"] = list(c[:EXEC])
                slots[w]["bounds"].append(len(slots[w]["acts"]))
        tm["policy"] += time.time() - tp
        tp = time.time()
        for w in live:
            s = slots[w]
            a = np.array(s["queue"].pop(0), dtype=np.float64)
            a[6] = 1.0 if a[6] > 0 else -1.0
            if ema and s["acts"]:
                a[:6] = ema * s["acts"][-1][:6] + (1 - ema) * a[:6]
            s["acts"].append(a)
            pipes[w].send(("step", a.tolist()))
        finished = []
        for w in live:
            s = slots[w]
            s["frame"], eef, done = pipes[w].recv()
            s["eef"].append(eef)
            if record:
                s["video"].append(s["frame"]["img"])
            if done or len(s["acts"]) >= MAX_STEPS:
                s["success"] = done
                done_eps.append(metrics(s))
                if record:  # both camera views per step, for make_videos.py
                    t, k = s["job"]
                    np.savez_compressed(os.path.join(record, f"t{t}_i{k}.npz"), video=np.stack(s["video"]), success=done,
                                        acts=np.array(s["acts"]), bounds=np.array(s["bounds"]))
                finished.append(w)
        tm["step"] += time.time() - tp
        if finished:
            tp = time.time()
            reset(finished)
            tm["reset"] += time.time() - tp
    for p in pipes:
        p.send(("close",))
    for pr in procs:
        pr.wait(timeout=30)

    sr = np.array([e["success"] for e in done_eps], float)
    agg = lambda k: float(np.mean([e[k] for e in done_eps]))
    return dict(
        ckpt=ckpt, ema=ema, episodes=len(done_eps), success_rate=float(sr.mean()),
        per_task={t: float(np.mean([e["success"] for e in done_eps if e["task"] == t])) for t in range(10)},
        ee_jerk_rms=agg("ee_jerk_rms"), seam_jump=agg("seam_jump"), inner_jump=agg("inner_jump"),
        latency_ms=latency_ms, wall_s=round(time.time() - t0, 1), time_s={k: round(v, 1) for k, v in tm.items()},
        per_episode=done_eps,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--jobs", help="only these episodes, e.g. 0:0,3:0 (task:init)")
    p.add_argument("--record", help="save per-episode camera frames here instead of writing eval.json next to the ckpt")
    a = p.parse_args()
    jobs = [tuple(map(int, j.split(":"))) for j in a.jobs.split(",")] if a.jobs else None
    if a.record:
        os.makedirs(a.record, exist_ok=True)
    r = evaluate(a.ckpt, a.episodes, a.workers, a.ema, jobs, a.record)
    out = os.path.join(a.record, "eval.json") if a.record else         os.path.join(os.path.dirname(a.ckpt), f"eval{'_ema' + str(a.ema) if a.ema else ''}.json")
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "per_episode"}))
