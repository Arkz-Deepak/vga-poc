"""External baseline: lerobot/smolvla_base (450M) fine-tuned on the identical few-shot split.

python smolvla_baseline.py --shots 5 --seed 0 --out runs/smolvla_s5_seed0
python eval.py runs/smolvla_s5_seed0/model --episodes 20
"""
import argparse, json, math, os, time
import numpy as np, torch

from vga import PROC, pick_episodes

CHUNK = 50  # smolvla_base's pretrained chunk size
CAMS = ("observation.images.camera1", "observation.images.camera2")


def make_policy(path, stats=None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.input_features = {
        CAMS[0]: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        CAMS[1]: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(11,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.empty_cameras, cfg.device = 0, "cuda"
    return SmolVLAPolicy.from_pretrained(path, config=cfg)


def tokenize(vlm_name):
    from transformers import AutoTokenizer
    from common import suite
    tok = AutoTokenizer.from_pretrained(vlm_name)
    t = tok([suite().get_task(i).language + "\n" for i in range(10)], padding="longest", return_tensors="pt")
    return t["input_ids"].cuda(), t["attention_mask"].bool().cuda()


def batch_of(img, state, task, toks, action=None):
    """img uint8 [B,2,224,224,3] (cuda), state normalized [B,11] -> SmolVLA batch dict."""
    x = img.permute(0, 1, 4, 2, 3).float() / 255
    b = {CAMS[0]: x[:, 0], CAMS[1]: x[:, 1], "observation.state": state,
         "observation.language.tokens": toks[0][task], "observation.language.attention_mask": toks[1][task]}
    if action is not None:
        b["action"] = action
    return b


def load_split_imgs(shots, seed, chunk_len):
    """Identical few-shot split as vga.py, with raw images, z-scored state/action and chunk indices (shared with X-VLA)."""
    task, chunk, imgs, states, actions = [], [], [], [], []
    n = 0
    for tid, picks in pick_episodes(shots, seed).items():
        ep = np.load(os.path.join(PROC, f"t{tid}", "ep.npy"))
        img = np.load(os.path.join(PROC, f"t{tid}", "img.npy"), mmap_mode="r")
        st = np.load(os.path.join(PROC, f"t{tid}", "state.npy"))
        ac = np.load(os.path.join(PROC, f"t{tid}", "action.npy"))
        for e in picks:
            r = np.flatnonzero(ep == e)
            imgs.append(np.asarray(img[r[0]:r[-1] + 1]))
            states.append(st[r])
            actions.append(ac[r])
            chunk.append(n + np.minimum(np.arange(len(r))[:, None] + np.arange(chunk_len), len(r) - 1))
            task.append(np.full(len(r), tid))
            n += len(r)
    S, A = np.concatenate(states).astype(np.float32), np.concatenate(actions).astype(np.float32)
    stats = dict(s_mean=S.mean(0), s_std=S.std(0).clip(1e-3), a_mean=A.mean(0), a_std=A.std(0).clip(1e-3))
    stats["a_mean"][6], stats["a_std"][6] = 0.0, 1.0
    return dict(n=n, img=torch.from_numpy(np.concatenate(imgs)),  # CPU uint8, gathered per batch
                S=torch.from_numpy((S - stats["s_mean"]) / stats["s_std"]).cuda(),
                A=torch.from_numpy((A - stats["a_mean"]) / stats["a_std"]).cuda(),
                chunk=torch.from_numpy(np.concatenate(chunk)).cuda(),
                task=torch.from_numpy(np.concatenate(task)).cuda()), stats


def fit(policy, make_batch, D, stats, args, betas=(0.9, 0.95), weight_decay=1e-10):
    """AdamW + warmup/cosine on the policy's trainable params; saves model, stats and log under args.out."""
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    print(f"frames {D['n']}, trainable params {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=betas, eps=1e-8, weight_decay=weight_decay)
    warm = min(1000, args.steps // 10)
    t0, log = time.time(), []
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1, step / warm) * (0.5 * (1 + math.cos(math.pi * step / args.steps)) * 0.975 + 0.025)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):  # effective batch = batch * accum (large batches don't fit in 8 GB)
            i = torch.randint(0, D["n"], (args.batch,))
            ic = i.cuda()
            b = make_batch(D["img"][i].cuda(non_blocking=True), D["S"][ic], D["task"][ic], D["A"][D["chunk"][ic]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(b)
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if step % 200 == 0 or step == args.steps:
            log.append(dict(step=step, loss=loss.item(), sec=round(time.time() - t0, 1)))
            print(json.dumps(log[-1]), flush=True)
    policy.save_pretrained(os.path.join(args.out, "model"))
    np.savez(os.path.join(args.out, "stats.npz"), **stats)
    json.dump(dict(vars(args), log=log), open(os.path.join(args.out, "train_log.json"), "w"))


def train(args):
    torch.manual_seed(args.seed)
    D, stats = load_split_imgs(args.shots, args.seed, CHUNK)
    policy = make_policy("lerobot/smolvla_base")
    toks = tokenize(policy.config.vlm_model_name)
    fit(policy, lambda img, s, t, a: batch_of(img, s, t, toks, a), D, stats, args)


class SmolVLAAgent:
    """Same call signature as eval.Policy: (frames, task ids, action histories) -> [B, CHUNK, 7] actions."""

    def __init__(self, model_dir):
        self.policy = make_policy(model_dir).eval()
        st = np.load(os.path.join(os.path.dirname(model_dir), "stats.npz"))
        self.s = {k: torch.from_numpy(st[k]).cuda() for k in st.files}
        self.toks = tokenize(self.policy.config.vlm_model_name)

    @torch.no_grad()
    def __call__(self, frames, tids, histories):
        img = torch.from_numpy(np.stack([f["img"] for f in frames])).cuda()
        state = (torch.from_numpy(np.stack([f["state"] for f in frames])).cuda() - self.s["s_mean"]) / self.s["s_std"]
        b = batch_of(img, state, torch.tensor(tids, device="cuda"), self.toks)
        self.policy.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.policy.predict_action_chunk(b)
        return (a.float() * self.s["a_std"] + self.s["a_mean"]).cpu().numpy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", required=True)
    train(p.parse_args())
