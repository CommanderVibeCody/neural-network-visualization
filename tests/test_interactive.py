"""Headless test of the INTERACTIVE code paths (no --auto script).

pygame.event.get is replaced by a scripted feed of synthetic mouse/key events, so the
real handlers and main loops run exactly as for a user. Catches bugs like state that
is only initialized in auto mode.

Run:  python tests/test_interactive.py      (or: python -m pytest tests)
"""
import os
import sys
import types

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame  # noqa: E402

import nn_visualizer as nv  # noqa: E402


def make_args(**kw):
    a = dict(mode="menu", auto_target="dataset", dataset="Iris", epochs=30, auto=False, seconds=1,
             screenshot_dir="screenshots", split="holdout", k=3, train_pct=70, val_pct=15, test_pct=15,
             test_pct_kfold=20, seed=1, stratified=True, show_test_curve=False, track="Oval",
             population=12, gen_time=5, generations=1)
    a.update(kw)
    return types.SimpleNamespace(**a)


class Feed:
    """Scripted event source: {frame_index: [events]}; posts QUIT after max_frames."""

    def __init__(self, script, max_frames):
        self.script, self.max_frames, self.frame = script, max_frames, 0
        self._orig = pygame.event.get

    def __call__(self, *a, **k):
        self._orig()  # drain the real queue
        evs = list(self.script.get(self.frame, []))
        self.frame += 1
        if self.frame > self.max_frames:
            evs.append(pygame.event.Event(pygame.QUIT))
        return evs

    def __enter__(self):
        pygame.event.get = self
        return self

    def __exit__(self, *exc):
        pygame.event.get = self._orig


def click(x, y):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x, y))


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="")


def test_menu_to_evolution_interactive():
    pygame.init()
    args = make_args()
    # Menu: click the "Live Evolution Training" button (right card)
    with Feed({2: [click(nv.W // 2 + 220, 350)]}, 50):
        choice = nv.run_menu(args)
    assert choice == "evolution", choice

    from evolution.app import EvoApp
    app = EvoApp(args)
    script = {}
    # Wizard steps via ENTER (frames 3, 6, 9), then interact with the running sim
    script[3] = [key(pygame.K_RETURN)]
    script[6] = [key(pygame.K_RETURN)]
    script[9] = [key(pygame.K_RETURN)]          # -> Run step, starts the simulation
    script[20] = [key(pygame.K_PLUS)]            # faster
    script[30] = [key(pygame.K_SPACE)]           # pause
    script[35] = [key(pygame.K_SPACE)]           # resume
    script[40] = [key(pygame.K_f)]               # fast-forward
    script[120] = [key(pygame.K_f)]              # back to rendering
    script[125] = [key(pygame.K_s)]              # save best genome (may be "nothing yet")
    script[130] = [key(pygame.K_MINUS)]
    with Feed(script, 160):
        res = app.run()
    assert res == "quit"
    assert app.step == 3 and app.sim is not None
    assert app.gen >= 2, f"expected at least one generation turnover, gen={app.gen}"
    if os.path.exists("best_genome.npz"):
        os.remove("best_genome.npz")


def test_evolution_back_to_menu_before_start():
    """Pressing Menu on the first wizard step must not touch sim state."""
    pygame.init()
    from evolution.app import EvoApp
    app = EvoApp(make_args())
    with Feed({3: [click(80, nv.H - 40)]}, 30):   # "< Menu" button
        res = app.run()
    assert res == "menu"


def test_menu_to_dataset_interactive():
    pygame.init()
    args = make_args()
    with Feed({2: [click(nv.W // 2 - 220, 350)]}, 50):
        choice = nv.run_menu(args)
    assert choice == "dataset", choice
    app = nv.App(args)
    script = {i * 3: [key(pygame.K_RETURN)] for i in range(1, 6)}  # walk wizard to Train
    script[25] = [key(pygame.K_SPACE)]
    script[27] = [key(pygame.K_SPACE)]
    script[30] = [key(pygame.K_n)]
    script[32] = [key(pygame.K_l)]
    with Feed(script, 70):
        res = app.run()
    assert res == "quit"
    assert app.train_state is not None and app.step in (5, 6)
    assert app.train_state["epoch"] > 0 or app.step == 6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name, flush=True)
    pygame.quit()
