# evaluate.py (özet)
from __future__ import annotations
import argparse, os
import numpy as np
try:
    from .env import SnakeEnv
    from .agent import QTable
    from .utils import save_json
except ImportError:
    from env import SnakeEnv
    from agent import QTable
    from utils import save_json

def evaluate(Q: QTable, episodes=50, n=10, seed=123, max_steps=500,
             render=False, fps=20, frame_skip=2):
    env = SnakeEnv(n=n, seed=seed, render=render, fps=fps, frame_skip=frame_skip,
                   title="Snake (evaluation)")
    totals = []
    for _ in range(episodes):
        s = env.reset(); steps = score = 0; done = False
        while not done and steps < max_steps:
            a = int(np.argmax(Q.Q[s])) if s in Q.Q else 1
            s, r, done, info = env.step(a)
            if info.get("ate"): score += 1
            steps += 1
        totals.append(score)
    env.close()
    arr = np.array(totals, dtype=float)
    return {
        "episodes": int(episodes),
        "grid": int(n),
        "mean_score": float(arr.mean()),
        "std_score": float(arr.std()),
        "max_score": int(arr.max()),
        "min_score": int(arr.min()),
    }

def main():
    ap = argparse.ArgumentParser(description="Evaluate a saved Q-table on Snake")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    Q = QTable.load(args.checkpoint)
    stats = evaluate(Q, episodes=args.episodes, n=args.grid, seed=args.seed,
                     max_steps=args.max_steps, render=args.render,
                     fps=args.fps, frame_skip=args.frame_skip)
    out = args.out or os.path.join(os.path.dirname(args.checkpoint), "eval.json")
    save_json(out, stats)
    print(f"[eval] {out} -> mean={stats['mean_score']:.2f}, max={stats['max_score']}")
