"""Compact VLA ("VGA-lite") with switchable add-ons, trained few-shot on LIBERO-Spatial.

Base (SmolVLA-style): frozen SigLIP image/text features -> small transformer trunk -> flow-matching action expert.
  --depth   training-only quantized-depth head on mid-trunk visual tokens (detached at test time)
  --rays    per-token Plücker camera-ray embedding (world frame; wrist-cam rays move with the arm)
  --smooth  previous-action prefix conditioning + acceleration/jerk penalty across the chunk seam

python vga.py --shots 5 --seed 0 [--depth] [--rays] [--smooth] --out runs/x
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "data", "proc")
H, P, NBINS = 16, 4, 32
DEPTH_EDGES = torch.logspace(math.log10(0.03), math.log10(3.5), NBINS - 1)  # metres


class VGA(nn.Module):
    def __init__(self, depth=False, rays=False, smooth=False, d=384, heads=6, L_trunk=4, L_exp=4):
        super().__init__()
        self.use_depth, self.use_rays, self.use_smooth = depth, rays, smooth
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.vis_in = nn.Linear(768, d)
        self.cam_pos = nn.Parameter(torch.randn(2, 49, d) * 0.02)
        self.ray_in = nn.Linear(6, d) if rays else None
        self.lang_in, self.state_in = nn.Linear(768, d), nn.Linear(11, d)
        self.trunk = nn.ModuleList(layer() for _ in range(L_trunk))
        self.trunk_norm = nn.LayerNorm(d)
        self.depth_head = nn.Linear(d, 16 * NBINS) if depth else None  # 4x4 sub-patch depth bins per token
        self.act_in, self.act_out = nn.Linear(7, d), nn.Linear(d, 7)
        self.act_pos = nn.Parameter(torch.randn(P + H, d) * 0.02)
        self.prefix_type = nn.Parameter(torch.randn(d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.expert = nn.TransformerDecoder(dec, L_exp)
        self.d = d

    def encode(self, feat, ray, lang, state):
        """feat [B,2,49,768], ray [B,2,49,6], lang [B,768], state [B,11] -> context [B,100,d], mid-layer visual tokens."""
        v = self.vis_in(feat) + self.cam_pos
        if self.use_rays:
            v = v + self.ray_in(ray)
        x = torch.cat([v.flatten(1, 2), self.lang_in(lang)[:, None], self.state_in(state)[:, None]], 1)
        mid = None
        for i, blk in enumerate(self.trunk):
            x = blk(x)
            if i == len(self.trunk) - 2:  # "deep but not deepest" (Spatial Forcing)
                mid = x[:, :98]
        return self.trunk_norm(x), mid

    def velocity(self, ctx, x_t, t, prefix, prefix_mask):
        te = self.t_mlp(sinusoid(t, self.d))[:, None]
        a = self.act_in(x_t) + self.act_pos[P:] + te
        if self.use_smooth:
            p = (self.act_in(prefix) + self.prefix_type + self.act_pos[:P]) * prefix_mask[..., None]
            a = torch.cat([p, a], 1)
        return self.act_out(self.expert(a, ctx))[:, -H:]

    @torch.no_grad()
    def sample(self, ctx, prefix, prefix_mask, steps=10, x=None):
        x = torch.randn(ctx.shape[0], H, 7, device=ctx.device) if x is None else x
        for i in range(steps):
            t = torch.full((ctx.shape[0],), i / steps, device=ctx.device)
            x = x + self.velocity(ctx, x, t, prefix, prefix_mask) / steps
        return x


def sinusoid(t, d):
    f = torch.exp(-math.log(10000.0) * torch.arange(d // 2, device=t.device) / (d // 2))
    a = (t[:, None] * 1000.0) * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


def pick_episodes(shots, seed):
    """{task: sorted demo ids} -- the few-shot split; shared with the SmolVLA baseline."""
    rng = np.random.default_rng(seed)
    out = {}
    for tid in range(10):
        ep = np.load(os.path.join(PROC, f"t{tid}", "ep.npy"))
        out[tid] = sorted(rng.permutation(np.unique(ep))[:shots])
    return out


def load_split(shots, seed, device):
    """Load the few-shot split's cached features; build chunk/prefix indices."""
    lang_all = np.load(os.path.join(PROC, "lang.npy"))
    keys = ("feat", "ray", "depth", "state", "action")
    parts = {k: [] for k in keys}
    task, chunk_idx, prefix_idx, prefix_mask = [], [], [], []
    n = 0
    for tid, picks in pick_episodes(shots, seed).items():
        ep = np.load(os.path.join(PROC, f"t{tid}", "ep.npy"))
        arrs = {k: np.load(os.path.join(PROC, f"t{tid}", f"{k}.npy"), mmap_mode="r") for k in keys}
        for e in picks:
            rows = np.flatnonzero(ep == e)
            T = len(rows)
            for k in keys:
                parts[k].append(np.asarray(arrs[k][rows[0]:rows[-1] + 1]))
            i = np.arange(T)
            chunk_idx.append(n + np.minimum(i[:, None] + np.arange(H), T - 1))
            pi = i[:, None] + np.arange(-P, 0)
            prefix_idx.append(n + np.maximum(pi, 0))
            prefix_mask.append(pi >= 0)
            task.append(np.full(T, tid))
            n += T
    D = {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}
    a = D["action"].float()
    stats = dict(a_mean=a.mean(0), a_std=a.std(0).clamp_min(1e-3), s_mean=D["state"].float().mean(0),
                 s_std=D["state"].float().std(0).clamp_min(1e-3))
    stats["a_mean"][6], stats["a_std"][6] = 0.0, 1.0  # gripper stays in {-1, 1}
    out = dict(
        feat=D["feat"].to(device),  # fp16, ~0.4 GB per 10 demos/task
        ray=D["ray"].float().flatten(2, 3).to(device),
        depth=bucket_depth(D["depth"].float()).to(device),
        state=((D["state"].float() - stats["s_mean"]) / stats["s_std"]).to(device),
        action=((a - stats["a_mean"]) / stats["a_std"]).to(device),
        lang=torch.from_numpy(lang_all).float().to(device)[torch.from_numpy(np.concatenate(task))],
        chunk=torch.from_numpy(np.concatenate(chunk_idx)).to(device),
        prefix=torch.from_numpy(np.concatenate(prefix_idx)).to(device),
        prefix_mask=torch.from_numpy(np.concatenate(prefix_mask)).float().to(device),
    )
    return out, stats


def bucket_depth(depth):
    """[N,2,28,28] metres -> [N,2,49,16] bin ids (4x4 sub-grid per 7x7 token)."""
    b = torch.bucketize(depth, DEPTH_EDGES)
    return b.view(-1, 2, 7, 4, 7, 4).permute(0, 1, 2, 4, 3, 5).reshape(-1, 2, 49, 16)


def seam_smoothness(seq, valid):
    """Mean squared accel and jerk of seq [B,T,6] over windows whose samples are all valid [B,T]."""
    acc = seq[:, 2:] - 2 * seq[:, 1:-1] + seq[:, :-2]
    jerk = acc[:, 1:] - acc[:, :-1]
    va = valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]
    vj = va[:, 1:] * va[:, :-1]
    return ((acc ** 2).mean(-1) * va).sum() / va.sum(), ((jerk ** 2).mean(-1) * vj).sum() / vj.sum()


def train(args):
    torch.manual_seed(args.seed)
    dev = "cuda"
    D, stats = load_split(args.shots, args.seed, dev)
    N = len(D["state"])
    model = VGA(args.depth, args.rays, args.smooth, d=args.d, heads=args.d // 64).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * s / args.steps))
    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step)
        i = torch.randint(0, N, (args.batch,), device=dev)
        x1 = D["action"][D["chunk"][i]]
        prefix, pmask = D["action"][D["prefix"][i]], D["prefix_mask"][i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx, mid = model.encode(D["feat"][i].float(), D["ray"][i], D["lang"][i], D["state"][i])
            t = torch.rand(len(i), device=dev)
            x0 = torch.randn_like(x1)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
            v = model.velocity(ctx, xt, t, prefix, pmask)
        v = v.float()
        loss_flow = F.mse_loss(v, x1 - x0)
        loss = loss_flow
        logs = {"flow": loss_flow.item()}
        if args.depth:
            logits = model.depth_head(mid.float()).view(len(i), 98, 16, NBINS)
            loss_d = F.cross_entropy(logits.reshape(-1, NBINS), D["depth"][i].reshape(-1))
            loss = loss + args.w_depth * loss_d
            logs["depth"] = loss_d.item()
        if args.smooth:
            x1_hat = xt + (1 - t[:, None, None]) * v  # clean-chunk estimate
            seq = torch.cat([prefix[..., :6], x1_hat[..., :6]], 1)
            valid = torch.cat([pmask, torch.ones(len(i), H, device=dev)], 1) * torch.cat(
                [torch.ones(len(i), P, device=dev), t[:, None].expand(-1, H)], 1).gt(0.3).float()
            l_acc, l_jerk = seam_smoothness(seq, valid)
            lam = args.w_kin * min(1.0, step / (0.5 * args.steps))
            loss = loss + lam * (l_acc + 0.5 * l_jerk)
            logs.update(acc=l_acc.item(), jerk=l_jerk.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == args.steps:
            logs.update(step=step, sec=round(time.time() - t0, 1))
            log.append(logs)
            print(json.dumps(logs), flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))
    json.dump(log, open(os.path.join(args.out, "train_log.json"), "w"))


def load_policy(path, dev="cuda"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    m = VGA(a["depth"], a["rays"], a["smooth"], d=a["d"], heads=a["d"] // 64).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, {k: v.to(dev) for k, v in ck["stats"].items()}, a


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth", action="store_true")
    p.add_argument("--rays", action="store_true")
    p.add_argument("--smooth", action="store_true")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w_depth", type=float, default=0.1)
    p.add_argument("--w_kin", type=float, default=0.05)
    p.add_argument("--out", required=True)
    train(p.parse_args())
