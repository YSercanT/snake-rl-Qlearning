from __future__ import annotations
import random
import pickle
from collections import defaultdict
from typing import Any, Dict
import numpy as np
from pathlib import Path

class QTable:
    """Q-learning agent (ε-greedy)."""

    def __init__(self, actions: int = 3):
        self.actions = actions
        self.Q: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(actions, dtype=np.float32)
        )

    def act(self, state, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.actions)
        q = self.Q[state]
        m = float(np.max(q))
        idxs = [i for i, v in enumerate(q) if v == m]
        return random.choice(idxs)

    def update(self, s, a: int, r: float, s2, alpha: float = 0.3, gamma: float = 0.99):
        qsa = self.Q[s][a]
        td = r + gamma * float(np.max(self.Q[s2])) - qsa
        self.Q[s][a] = qsa + alpha * td

    def greedy(self, state) -> int:
        q = self.Q[state]
        return int(np.argmax(q))

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)        
        with p.open("wb") as f:
            pickle.dump({"actions": self.actions, "Q": dict(self.Q)}, f)
        size = p.stat().st_size
        print(f"[checkpoint] wrote {p} ({size} bytes)")
        return str(p)

    @classmethod
    def load(cls, path: str) -> "QTable":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        agent = cls(actions=obj["actions"])
        agent.Q = defaultdict(lambda: np.zeros(agent.actions, dtype=np.float32))
        for k, v in obj["Q"].items():
            agent.Q[k] = np.array(v, dtype=np.float32)
        return agent
