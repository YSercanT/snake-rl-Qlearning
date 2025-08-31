from __future__ import annotations
import argparse, os, random
from typing import Tuple
import numpy as np

try:
    from .env import SnakeEnv
    from .agent import QTable
    from .utils import make_run_dir, save_json, save_csv, plot_scores
except ImportError:
    from env import SnakeEnv
    from agent import QTable
    from utils import make_run_dir, save_json, save_csv, plot_scores

def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)

def train(
    episodes: int = 2500,   # preset for 12x12
    n: int = 12,
    alpha: float = 0.05,
    gamma: float = 0.90,
    eps_start: float = 1.0,
    eps_end: float = 0.001,
    max_steps: int = 600,
    seed: int = 0,
    render: bool = False,
    fps: int = 20,
    frame_skip: int = 2,
) -> Tuple[QTable, list[int], dict]:
    """Main training loop. Frame-skipped rendering keeps it watchable & fast."""
    set_seeds(seed)
    env = SnakeEnv(n=n, seed=seed, render=render, fps=fps, frame_skip=frame_skip,
                   title="Snake Q-learning (training)")
    Q = QTable(actions=3)
    scores: list[int] = []
    best = {"score": -1, "episode": -1}
    eps_decay = max(1, episodes)

    for ep in range(1, episodes + 1):
        s = env.reset()
        score = steps = 0; done = False
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / eps_decay))
        while not done and steps < max_steps:
            a = Q.act(s, eps)
            s2, r, done, info = env.step(a)
            Q.update(s, a, r, s2, alpha=alpha, gamma=gamma)
            s = s2
            if info.get("ate"): score += 1
            steps += 1
        scores.append(score)
        if score > best["score"]: best = {"score": score, "episode": ep}
        if ep % 200 == 0:
            avg = float(np.mean(scores[-200:]))
            print(f"Ep {ep:4d} | avg(last200)={avg:.2f} | best={best['score']}@{best['episode']}")
    env.close()

    # summary metrics
    metrics = {
        "episodes": int(episodes),
        "grid": int(n),
        "alpha": float(alpha), "gamma": float(gamma),
        "eps_start": float(eps_start), "eps_end": float(eps_end),
        "max_steps": int(max_steps), "seed": int(seed),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "last100_mean": float(np.mean(scores[-100:])) if scores else 0.0,
        "best_score": int(best["score"]), "best_episode": int(best["episode"]),
    }
    return Q, scores, metrics

def main():
    ap = argparse.ArgumentParser(description="Train Q-learning agent for Snake")
  
    ap.add_argument("--episodes",  type=int,   default=2500)
    ap.add_argument("--grid",      type=int,   default=12)
    ap.add_argument("--alpha",     type=float, default=0.05)
    ap.add_argument("--gamma",     type=float, default=0.90)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end",   type=float, default=0.001)
    ap.add_argument("--max-steps", type=int,   default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame-skip", type=int, default=2)

    ap.add_argument("--runs-root", type=str, default="artifacts/runs")
    ap.add_argument("--tag", type=str, default=None, help="append a short tag to run folder")

    #  explicit paths 
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--log-csv", type=str, default=None)
    args = ap.parse_args()

    # create run directory
    run_dir = make_run_dir(args.runs_root, args.tag)
    ckpt = args.checkpoint or os.path.join(run_dir, "qtable.pkl")
    log_csv = args.log_csv or os.path.join(run_dir, "scores.csv")
    plot_png = os.path.join(run_dir, "score_plot.png")
    config_json = os.path.join(run_dir, "config.json")
    metrics_json = os.path.join(run_dir, "metrics.json")

    # save config  
    config = {
        "episodes": args.episodes, "grid": args.grid,
        "alpha": args.alpha, "gamma": args.gamma,
        "eps_start": args.eps_start, "eps_end": args.eps_end,
        "max_steps": args.max_steps, "seed": args.seed,
        "render": bool(args.render), "fps": args.fps, "frame_skip": args.frame_skip,
        "run_dir": run_dir,
    }
    save_json(config_json, config)

    # train
    Q, scores, metrics = train(
        episodes=args.episodes, n=args.grid,
        alpha=args.alpha, gamma=args.gamma,
        eps_start=args.eps_start, eps_end=args.eps_end,
        max_steps=args.max_steps, seed=args.seed,
        render=args.render, fps=args.fps, frame_skip=args.frame_skip,
    )

    ckpt_run = os.path.join(run_dir, "qtable.pkl")
    Q.save(ckpt_run)
    save_csv(log_csv, header=["episode", "score"], rows=((i+1, s) for i, s in enumerate(scores)))
    save_json(metrics_json, metrics)
    plot_scores(scores, path=plot_png, window=100)

    print(f"[run-dir]   {run_dir}")
    print(f"[checkpoint]{ckpt_run}")
    print(f"[scores]    {log_csv}")
    print(f"[metrics]   {metrics_json}")
    print(f"[plot]      {plot_png} (if matplotlib installed)")

