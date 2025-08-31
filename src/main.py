from __future__ import annotations
import argparse, random, os
from pathlib import Path

try:
    from .env import SnakeEnv
    from .train import train
    from .evaluate import evaluate as eval_fn
    from .agent import QTable
    from .utils import make_run_dir, save_json, save_csv, plot_scores, latest_run_dir
except ImportError:
    from env import SnakeEnv
    from train import train
    from evaluate import evaluate as eval_fn
    from agent import QTable
    from utils import make_run_dir, save_json, save_csv, plot_scores, latest_run_dir

ROOT = Path(__file__).resolve().parents[1]
RUNS_BASE = ROOT / "artifacts" / "runs"

def main():
    ap = argparse.ArgumentParser(prog="python -m src.main", add_help=True)
    ap.add_argument("--mode", choices=["demo","train","eval"], default="demo")
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--out", type=str, default=None)     
    args = ap.parse_args()

    if args.mode == "demo":
        env = SnakeEnv(n=args.grid, render=True, fps=args.fps, frame_skip=args.frame_skip, title="Snake Demo")
        s = env.reset(); done=False; steps=0
        while not done and steps < 600:
            s, r, done, info = env.step(random.randrange(3))
            steps += 1
        env.close()
        return

    if args.mode == "train":
        run_dir = Path(make_run_dir(str(RUNS_BASE), args.tag))
        Q, scores, metrics = train(
            episodes=args.episodes, n=args.grid,
            seed=args.seed, render=args.render, fps=args.fps, frame_skip=args.frame_skip,
        )
        ckpt = run_dir / "qtable.pkl"            # save after training
        Q.save(str(ckpt))
        save_csv(run_dir / "scores.csv", ["episode","score"], ((i+1,s) for i,s in enumerate(scores)))
        save_json(run_dir / "metrics.json", metrics)
        plot_scores(scores, run_dir / "score_plot.png")
        print(f"[run-dir]   {run_dir}")
        print(f"[checkpoint]{ckpt}")
        return

    if args.mode == "eval":
        run_dir = Path(latest_run_dir(str(RUNS_BASE))) if args.latest or True else None
        if not run_dir:
            raise SystemExit(f"No runs found under {RUNS_BASE}. Train first.")
        ckpt = run_dir / "qtable.pkl"
        Q = QTable.load(str(ckpt))
        stats = eval_fn(
            Q, episodes=args.episodes, n=args.grid, seed=args.seed,
            max_steps=500, render=args.render, fps=args.fps, frame_skip=args.frame_skip,
        )
        out = Path(args.out) if args.out else (run_dir / "eval.json")
        save_json(out, stats)
        print(f"[eval] {out} -> mean={stats['mean_score']:.2f}, max={stats['max_score']}")
        return

if __name__ == "__main__":
    main()
