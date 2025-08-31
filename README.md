# Snake AI with Q-Learning (Pygame)

This repository implements a **tabular Q-learning** agent that plays **Snake** with **real-time Pygame rendering enabled by default**.  
The codebase is compact and readable: training and evaluation scripts are provided, and all outputs are saved under `artifacts/runs/<timestamp>-<tag>/`.


---

## Features

- **State (compact & discrete)**  
  `(danger_left, danger_front, danger_right, sdx, sdy, dir)`  
  - `danger_* ∈ {0,1}` — collision if we move left/forward/right  
  - `sdx, sdy ∈ {-1,0,1}` — food direction along X/Y  
  - `dir ∈ {0..3}` — heading (U,R,D,L)
- **Actions:** `0 = turn left`, `1 = straight`, `2 = turn right`
- **Reward shaping (from `env.py`):**
  - `+10` eat food
  - `−10` die (wall/body)
  - `−0.01` per step
  - `+0.05` when the head gets **closer** to food; `−0.02` when it gets **farther**
- **Live rendering:** Pygame grid, snake, food, and a tiny HUD (length/steps)

---

## Install

```bash
# (Optional) virtual env
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1 or .\.venv\Scripts\Activate.bat

# Dependencies
pip install -r requirements.txt
# requirements.txt: numpy, pygame ,matplotlib

```

## Quickstart

> Run from the **project root** (same folder as this README).

### Train (with live rendering)

```bash
python -m src.main --mode train --render --tag exp1 --grid 12 --episodes 2500
```

### Evaluate (loads the latest run automatically)

```bash
python -m src.main --mode eval --render --grid 12 --episodes 100
```

### Demo (random policy)
```bash
python -m src.main --mode demo --render --grid 12 
```

## Outputs

Each training/eval run creates a timestamped folder:

```
artifacts/
└── runs/
    └── 2025MMDD-HHMMSS-<tag>/
        ├── qtable.pkl       # learned Q-table
        ├── scores.csv       # (episode, score)
        ├── metrics.json     # mean/std/best/last100, etc.
        ├── score_plot.png   # training curve
        └── eval.json        # written during --mode eval
```

---

## CLI Reference

### `python -m src.main`

Main entrypoint for **demo/train/eval**.

**Modes**

* **demo** — Runs a random policy (no learning). Used this to smoke‑test the window/renderer, tweak --fps / --frame-skip, or capture quick visuals.

* **train** — Trains the Q‑learning agent. If --render is provided, episodes are visualized. Artifacts are saved under artifacts/runs/<timestamp>-<tag>/ (e.g., qtable.pkl, scores.csv, metrics.json, score_plot.png).

* **eval** — Evaluates a saved model with a greedy policy (argmax). Loads the latest run automatically and writes eval.json next to the checkpoint. You can also evaluate a specific checkpoint via python -m src.evaluate --checkpoint ....


| Argument       | Type                        | Default | Purpose                                 |
| -------------- | --------------------------- | ------- | --------------------------------------- |
| `--mode`       | `demo` \| `train` \| `eval` | `demo`  | Select behavior                         |
| `--grid`       | `int`                       | `12`    | Board size (e.g., `12`)                 |
| `--episodes`   | `int`                       | `2500`  | Number of episodes                      |
| `--seed`       | `int`                       | `0`     | RNG seed                                |
| `--render`     | flag                        | `False` | Open the Pygame window                  |
| `--fps`        | `int`                       | `20`    | Render FPS                              |
| `--frame-skip` | `int`                       | `2`     | Draw every Nth frame (speeds up render) |
| `--tag`        | `str`                       | `main`  | Suffix for the run folder name          |
| `--out`        | `str`                       | `None`  | Custom path for `eval.json` (eval mode) |

> Recommended 12×12 settings are shown in **Quickstart** commands (explicit flags).

### (Advanced) `python -m src.train`

Call the training module directly to tweak **learning hyperparameters**:

Common flags:

* `--alpha` (learning rate)
* `--gamma` (discount)
* `--eps-start`, `--eps-end` (ε-greedy)
* `--max-steps` (per episode)
* `--tag` (run folder suffix)

**Example (recommended 12×12 preset):**

```bash
python -m src.train --grid 12 --episodes 2500 \
  --alpha 0.05 --gamma 0.90 \
  --eps-start 1.0 --eps-end 0.001 \
  --max-steps 600 --tag preset12
```

**Evaluate a specific checkpoint:**

```bash
python -m src.evaluate --checkpoint artifacts/runs/<RUN_ID>/qtable.pkl \
  --grid 12 --episodes 100 --render
```

---

## How It Works (short)

* **Agent (`agent.py`)** — ε-greedy **tabular Q-learning**
  $Q(s,a) ← Q(s,a) + α ( r + γ max_{a'} Q(s',a') − Q(s,a) )$
* **Environment (`env.py`)** — small custom snake grid with food; exposes a compact discrete state; renders via Pygame.
* **Training (`train.py`)** — runs episodes, logs `scores.csv` / `metrics.json`, saves `qtable.pkl`, `score_plot.png`.
* **Evaluation (`evaluate.py`)** — runs a greedy policy, writes `eval.json` (mean/std/min/max).

---

## Project Structure

```
src/
  __init__.py
  agent.py       # QTable (ε-greedy, update, save/load)
  env.py         # SnakeEnv (logic + Pygame rendering)
  train.py       # training loop + logging/plots
  evaluate.py    # greedy evaluation -> eval.json
  utils.py       # run folder + I/O helpers 
artifacts/
  runs/          # per-run outputs (auto-created)
requirements.txt
README.md
```

