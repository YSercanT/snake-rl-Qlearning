from __future__ import annotations
import os, json, time
from typing import Iterable, Sequence, Any, Dict
import matplotlib.pyplot as plt

def timestamp_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def make_run_dir(base: str = "artifacts/runs", tag: str | None = None) -> str:
    run_id = timestamp_id() + (f"-{tag}" if tag else "")
    path = os.path.join(base, run_id)
    os.makedirs(path, exist_ok=True)
    return path

def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_csv(path: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

def moving_average(xs: Sequence[float], window: int) -> list[float]:
    w = max(1, int(window))
    out, acc = [], 0.0
    q = []
    for x in xs:
        q.append(float(x)); acc += float(x)
        if len(q) > w:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out

def plot_scores(scores: Sequence[float], path: str, window: int = 100) -> None:
    xs = list(range(1, len(scores)+1))
    plt.figure(figsize=(7.5, 4.0))
    plt.plot(xs, scores, label="score")
    if len(scores) > 2:
        ma = moving_average(scores, window)
        plt.plot(xs, ma, label=f"moving avg ({window})")
    plt.xlabel("episode"); plt.ylabel("score"); plt.title("Training scores")
    plt.legend(); plt.tight_layout()
    plt.savefig(path)
    plt.close()
def latest_run_dir(base: str = "artifacts/runs") -> str | None:
    if not os.path.isdir(base):
        return None
    subdirs = [
        os.path.join(base, d)
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ]
    return max(subdirs, key=os.path.getmtime) if subdirs else None