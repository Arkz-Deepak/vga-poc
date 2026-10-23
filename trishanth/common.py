"""LIBERO helpers shared by data prep and evaluation."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(ROOT, ".libero"))

import numpy as np
import robosuite.utils.transform_utils as T
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv  # torch-free; benchmark (torch) is imported lazily
from robosuite.utils.camera_utils import get_real_depth_map

SUITE = "libero_spatial"
CAMS = ("agentview", "robot0_eye_in_hand")
RES = 224   # render size == SigLIP input size
GRID = 7    # visual tokens per side (SigLIP 14x14 patches, 2x2 pooled)
NOOP = [0, 0, 0, 0, 0, 0, -1]


def suite():
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[SUITE]()


def bddl_path(task_id):
    t = suite().get_task(task_id)
    return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)


def make_env(bddl, depth=False):
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES, camera_depths=depth,
                             hard_reset=False)  # soft reset: pixel-identical, 2.5x faster
    env.seed(0)
    return env


def proprio(obs):
    """eef pos (3) + continuous 6D rotation (3x2 columns of R) + gripper qpos (2)."""
    R = T.quat2mat(obs["robot0_eef_quat"])
    return np.concatenate([obs["robot0_eef_pos"], R[:, :2].T.reshape(-1), obs["robot0_gripper_qpos"]]).astype(np.float32)


def cam_geom(sim, cam):
    cid = sim.model.camera_name2id(cam)
    f = 0.5 * RES / np.tan(np.deg2rad(sim.model.cam_fovy[cid]) / 2)
    return f, sim.data.cam_xpos[cid].copy(), sim.data.cam_xmat[cid].reshape(3, 3).copy()


def rays(sim, cam):
    """Plücker rays (d, o x d), world frame, at the GRID x GRID token centres of the upright image."""
    f, o, R = cam_geom(sim, cam)
    c = (np.arange(GRID) + 0.5) * RES / GRID - RES / 2
    x, y = np.meshgrid(c / f, -c / f)  # image u right / v down -> camera x right / y up, looking down -z
    d = np.stack([x, y, -np.ones_like(x)], -1) @ R.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return np.concatenate([d, np.cross(o, d)], -1).astype(np.float32)  # [G, G, 6]


def project(sim, cam, p):
    """World point -> (u, v) pixel in the upright image. Used only to sanity-check `rays`."""
    f, o, R = cam_geom(sim, cam)
    pc = R.T @ (p - o)
    return f * pc[0] / -pc[2] + RES / 2, -f * pc[1] / -pc[2] + RES / 2


def frame(obs, sim, depth=False):
    out = dict(
        img=np.stack([obs[f"{c}_image"][::-1] for c in CAMS]),  # undo OpenGL vertical flip
        ray=np.stack([rays(sim, c) for c in CAMS]),
        state=proprio(obs),
    )
    if depth:
        out["depth"] = np.stack([get_real_depth_map(sim, obs[f"{c}_depth"])[::-1, :, 0] for c in CAMS]).astype(np.float32)
    return out


def worker_main(port):
    """Sim worker for eval.py. Runs as its own process so it never imports torch (~4 GB commit each on Windows)."""
    from multiprocessing.connection import Client
    conn = Client(("127.0.0.1", port), authkey=b"libero")
    env, cur = None, None
    while True:
        cmd, *arg = conn.recv()
        if cmd == "reset":
            bddl, init, settle = arg
            if bddl != cur:
                if env:
                    env.close()
                env, cur = make_env(bddl), bddl
            env.reset()
            obs = env.set_init_state(init)
            for _ in range(settle):
                obs, *_ = env.step(NOOP)
            conn.send((frame(obs, env.env.sim), obs["robot0_eef_pos"].copy(), False))
        elif cmd == "step":
            obs, _, done, _ = env.step(arg[0])
            conn.send((frame(obs, env.env.sim), obs["robot0_eef_pos"].copy(), bool(done)))
        else:
            if env:
                env.close()
            return


if __name__ == "__main__":
    import sys
    worker_main(int(sys.argv[1]))
