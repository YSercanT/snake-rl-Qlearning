from __future__ import annotations
import random
from typing import Tuple, Dict, Any, List

try:
    import pygame 
except Exception:
    pygame = None  

# Directions: U, R, D, L
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

BG_COLOR   = (30, 30, 36)
GRID_COLOR = (45, 45, 52)
FOOD_COLOR = (237, 85, 59)
HEAD_COLOR = (92, 167, 255)
BODY_COLOR = (64, 140, 230)


class SnakeEnv:
    """
    Compact env for tabular Q-learning: 
      - State: (danger_left, danger_front, danger_right, sdx[-1/0/1], sdy[-1/0/1], dir[0..3])
      - Action: 0=turn left, 1=straight, 2=turn right
      - Reward: +10 eat, -10 die, -0.01 step; +0.05 closer to food, -0.02 farther
   
    """

    # --- init
    def __init__(
        self,
        n: int = 12,
        seed: int | None = None,
        render: bool = False,
        cell_px: int = 32,
        fps: int = 15,
        frame_skip: int = 1,
        show_grid: bool = True,
        title: str = "Snake (Q-learning)"
    ):
        self.n = n
        self.rng = random.Random(seed)
        self.dir = 1
        self.snake: List[Tuple[int, int]] = []
        self.food: Tuple[int, int] = (0, 0)
        self.steps = 0
        self.prev_dist = 0

        # render config
        self.render_enabled = render
        self._frame_skip = max(1, int(frame_skip))
        self._draw_every = self._frame_skip
        self._cell = cell_px
        self._fps = fps
        self._show_grid = show_grid
        self._title = title

        # pygame members
        self._pg_ready = False
        self._screen = None
        self._clock = None
        self._font = None

    
    def reset(self):
        self.dir = 1  # start right
        mid = self.n // 2
        self.snake = [(mid, mid), (mid - 1, mid)]  # head first
        self._place_food()
        self.steps = 0
        self.prev_dist = self._manhattan(self.snake[0], self.food)
        if self.render_enabled and not self._pg_ready:
            self._init_pygame()
        return self._state()

    def step(self, action: int):
        # turn relative to current heading
        self._turn(action)

        # propose new head
        dx, dy = DIRS[self.dir]
        hx, hy = self.snake[0]
        new_head = (hx + dx, hy + dy)

        #death on wall
        if self._out_of_bounds(new_head):
            self._maybe_render()
            return self._state(), -10.0, True, {}

        # body collision check with tail exception
        will_eat = (new_head == self.food)
        body = self.snake if will_eat else self.snake[:-1]
        if new_head in body:
            self._maybe_render()
            return self._state(), -10.0, True, {}

        #move
        self.snake.insert(0, new_head)
        reward = -0.01
        info: Dict[str, Any] = {"ate": False}

        if will_eat:
            reward += 10.0
            info["ate"] = True
            self._place_food()
        else:
            self.snake.pop()  # tail moves

        #shaping: approach vs depart
        dist = self._manhattan(new_head, self.food)
        if dist < self.prev_dist:
            reward += 0.05
        elif dist > self.prev_dist:
            reward -= 0.02
        self.prev_dist = dist

        self.steps += 1

        # render (skipped frames for speed)
        self._maybe_render()

        return self._state(), reward, False, info

    def close(self):
        if self._pg_ready:
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._pg_ready = False
            self._screen = None
            self._clock = None
            self._font = None

    # state helpers
    def _state(self):
        hx, hy = self.snake[0]
        fx, fy = self.food
        sdx = 0 if fx == hx else (1 if fx > hx else -1)
        sdy = 0 if fy == hy else (1 if fy > hy else -1)
        return (
            self._danger(-1),
            self._danger(0),
            self._danger(+1),
            sdx,
            sdy,
            self.dir,
        )

    def _danger(self, rel: int) -> int:
        d = (self.dir + rel) % 4
        hx, hy = self.snake[0]
        nx, ny = hx + DIRS[d][0], hy + DIRS[d][1]
        return 1 if (self._out_of_bounds((nx, ny)) or (nx, ny) in self.snake[:-1]) else 0

    # geometry / RNG
    def _place_food(self):
        cells = {(x, y) for x in range(self.n) for y in range(self.n)} - set(self.snake)
        self.food = self.rng.choice(list(cells))

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _out_of_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return (x < 0 or x >= self.n or y < 0 or y >= self.n)

    def _turn(self, action: int):  # 0:left, 1:straight, 2:right
        if action == 0:
            self.dir = (self.dir - 1) % 4
        elif action == 2:
            self.dir = (self.dir + 1) % 4

    #  rendering 
    def _init_pygame(self):
        if pygame is None:
            raise RuntimeError("pygame not installed; add it to requirements.txt")
        pygame.init()
        flags = pygame.SCALED | pygame.RESIZABLE
        size = (self.n * self._cell, self.n * self._cell)
        self._screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption(self._title)
        self._clock = pygame.time.Clock()
        try:
            self._font = pygame.font.SysFont("consolas", 16)
        except Exception:
            self._font = None
        self._pg_ready = True

    def _maybe_render(self):
        if not self.render_enabled:
            return
        # frame skipping to keep training fast
        self._draw_every -= 1
        if self._draw_every > 0:
            return
        self._draw_every = self._frame_skip

        if not self._pg_ready:
            self._init_pygame()

        # handle basic events (quit window -> disable rendering, training continues headless)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.render_enabled = False
                return

        self._draw_frame()
        self._clock.tick(self._fps)

    def _draw_frame(self):
        scr = self._screen
        cell = self._cell
        w, h = scr.get_size()
        # dynamic cell size on resize
        cell = min(w // self.n, h // self.n)

        scr.fill(BG_COLOR)

        # grid (thin lines)
        if self._show_grid:
            for i in range(self.n + 1):
                x = i * cell
                y = i * cell
                pygame.draw.line(scr, GRID_COLOR, (x, 0), (x, self.n * cell), 1)
                pygame.draw.line(scr, GRID_COLOR, (0, y), (self.n * cell, y), 1)

        # food
        fx, fy = self.food
        pygame.draw.rect(
            scr, FOOD_COLOR, pygame.Rect(fx * cell + 1, fy * cell + 1, cell - 2, cell - 2), border_radius=4
        )

        # snake body
        for i, (x, y) in enumerate(self.snake):
            color = HEAD_COLOR if i == 0 else BODY_COLOR
            pygame.draw.rect(
                scr, color, pygame.Rect(x * cell + 2, y * cell + 2, cell - 4, cell - 4), border_radius=6
            )

        # HUD (tiny)
        if self._font:
            text = f"len={len(self.snake)}  steps={self.steps}"
            surf = self._font.render(text, True, (200, 200, 210))
            scr.blit(surf, (6, 6))

        pygame.display.flip()
if __name__ == "__main__":
    env = SnakeEnv(n=12, render=True, fps=20, frame_skip=2, title="Snake Demo (tmp)")
    s = env.reset()
    done = False
    steps = 0
    import random
    while not done and steps < 600:
        a = random.randrange(3)
        s, r, done, info = env.step(a)
        steps += 1
    env.close()