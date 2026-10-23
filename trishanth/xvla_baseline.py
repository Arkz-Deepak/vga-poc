"""External baseline: lerobot/xvla-base (X-VLA-0.9B) fine-tuned on the identical few-shot split.

Same recipe as smolvla_baseline.py: frozen vision-language backbone, trainable action transformer (+ soft prompts),
z-scored relative 7-D LIBERO actions via X-VLA's "auto" action mode, bf16 (LeRobot's recommended X-VLA precision).

python xvla_baseline.py --shots 5 --seed 0 --steps 8000 --batch 4 --accum 16 --out runs/xvla_eq_s5_seed0
python eval.py runs/xvla_eq_s5_seed0/model --episodes 20
"""
import argparse, os
import numpy as np, torch

from smolvla_baseline import fit, load_split_imgs

CHUNK = 30  # xvla-base's pretrained chunk size
CAMS = ("observation.images.image", "observation.images.image2")
DOMAIN = int(os.environ.get("XVLA_DOMAIN", 3))  # soft-prompt slot; lerobot/xvla-libero's LIBERO domain
IMAGENET = (torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))


def make_policy(path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.input_features = {
        CAMS[0]: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        CAMS[1]: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(11,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.action_mode = "auto"  # relative 7-D actions, like every other model in this study
    cfg.freeze_vision_encoder = cfg.freeze_language_encoder = True  # as SmolVLA: frozen VLM, train the action model
    cfg.dtype, cfg.device = "bfloat16", "cuda"
    return XVLAPolicy.from_pretrained(path, config=cfg)


def tokenize(name):
    from transformers import AutoTokenizer
    from common import suite
    tok = AutoTokenizer.from_pretrained(name)
    t = tok([suite().get_task(i).language for i in range(10)], padding="max_length", max_length=64,
            truncation=True, return_tensors="pt")
    return t["input_ids"].cuda()


def batch_of(img, state, task, toks, action=None):
    """img uint8 [B,2,224,224,3] (cuda), state normalized [B,11] -> X-VLA batch (ImageNet-normalized views)."""
    x = img.permute(0, 1, 4, 2, 3).float() / 255
    x = (x - IMAGENET[0].to(x.device)) / IMAGENET[1].to(x.device)
    b = {CAMS[0]: x[:, 0], CAMS[1]: x[:, 1], "observation.state": state,
         "observation.language.tokens": toks[task], "domain_id": torch.full((len(task),), DOMAIN, device=x.device)}
    if action is not None:
        b["action"] = action
    return b


def train(args):
    torch.manual_seed(args.seed)
    D, stats = load_split_imgs(args.shots, args.seed, CHUNK)
    policy = make_policy("lerobot/xvla-base")
    toks = tokenize(policy.config.tokenizer_name)
    fit(policy, lambda img, s, t, a: batch_of(img, s, t, toks, a), D, stats, args, betas=(0.9, 0.99), weight_decay=0.0)


class XVLAAgent:
    """Same call signature as eval.Policy: (frames, task ids, action histories) -> [B, CHUNK, 7] actions."""

    def __init__(self, model_dir):
        self.policy = make_policy(model_dir).eval()
        st = np.load(os.path.join(os.path.dirname(model_dir), "stats.npz"))
        self.s = {k: torch.from_numpy(st[k]).cuda() for k in st.files}
        self.toks = tokenize(self.policy.config.tokenizer_name)

    @torch.no_grad()
    def __call__(self, frames, tids, histories):
        img = torch.from_numpy(np.stack([f["img"] for f in frames])).cuda()
        state = (torch.from_numpy(np.stack([f["state"] for f in frames])).cuda() - self.s["s_mean"]) / self.s["s_std"]
        self.policy.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.policy.predict_action_chunk(batch_of(img, state, torch.tensor(tids, device="cuda"), self.toks))
        return (a.float() * self.s["a_std"] + self.s["a_mean"]).cpu().numpy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", required=True)
    train(p.parse_args())
