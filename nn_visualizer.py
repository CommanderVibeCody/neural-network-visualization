"""Neural Network Visualizer: Pygame wizard to configure a tf.keras network on a public
dataset (with hold-out or K-fold splits), then train while visualizing every layer live."""
import os
import sys
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pygame
import tensorflow as tf

from datasets import DATASETS, SplitConfig, load_dataset
from widgets import UI, BG, PANEL, TEXT, DIM, ACCENT, SEL

W, H = 1280, 780
ACTIVATIONS = ["relu", "tanh", "sigmoid", "elu", "leaky_relu", "selu", "linear"]
STEPS = ["Dataset", "Split", "Architecture", "Hidden", "Output", "Train", "Results"]
MAX_DRAWN = 16
COL_TRAIN, COL_VAL, COL_TEST = (60, 200, 90), (240, 160, 50), (220, 70, 70)


def output_options(task, n_classes):
    if task == "regression":
        return [("linear", 1, "mse")]
    if task == "binary":
        return [("sigmoid", 1, "binary_crossentropy"),
                ("softmax", 2, "sparse_categorical_crossentropy")]
    return [("softmax", n_classes, "sparse_categorical_crossentropy")]


class App:
    def __init__(self, args):
        self.args = args
        pygame.init()
        pygame.display.set_caption("Neural Network Visualizer (Keras + Pygame)")
        self.screen = pygame.display.set_mode((W, H))
        self.ui = UI(self.screen)
        self.med = pygame.font.SysFont("dejavusans", 17, bold=True)
        self.clock = pygame.time.Clock()
        self.step = 0
        self.ds_cache = {}
        self.error = ""
        self.split = SplitConfig(
            mode=args.split, train_pct=args.train_pct, val_pct=args.val_pct,
            test_pct=args.test_pct, k=args.k, test_pct_kfold=args.test_pct_kfold,
            stratified=args.stratified and True, seed=args.seed,
            show_test_curve=getattr(args, "show_test_curve", False))
        self.select_dataset(args.dataset)
        self.n_hidden = 2
        self.hidden = [{"units": 8, "act": "relu"}, {"units": 6, "act": "relu"}]
        self.selected = {0}
        self.model = None
        self.train_state = None
        self.log_scale = False  # L key toggles log y-axis on the error plot

    # ---------------- configuration ----------------
    def select_dataset(self, name):
        try:
            if name not in self.ds_cache:
                self.ds_cache[name] = load_dataset(name)
            self.ds = self.ds_cache[name]
            self.ds_name = name
            self.error = ""
        except Exception as e:
            self.error = f"Could not load {name}: {e}"
            if not hasattr(self, "ds"):
                self.select_dataset("Iris")
            return
        self.out_opts = output_options(self.ds.task, self.ds.n_classes)
        self.out_idx = 0
        if self.ds.task == "regression":
            self.split.stratified = False

    @property
    def out(self):
        return self.out_opts[self.out_idx]

    def set_n_hidden(self, n):
        n = max(0, min(8, n))
        while len(self.hidden) < n:
            self.hidden.append(dict(self.hidden[-1]) if self.hidden else {"units": 8, "act": "relu"})
        self.hidden = self.hidden[:n]
        self.n_hidden = n
        self.selected = {i for i in self.selected if i < n} or ({0} if n else set())

    def set_output_units(self, units):
        for i, o in enumerate(self.out_opts):
            if o[1] == units:
                self.out_idx = i

    def layer_sizes(self):
        return [self.ds.n_features] + [h["units"] for h in self.hidden] + [self.out[1]]

    def set_pct(self, which, v):
        """Adjust hold-out percentages; keep sum = 100 by borrowing from the other adjustable parts."""
        s = self.split
        v = max(5 if which != "test_pct" else 5, min(90, v))
        setattr(s, which, v)
        total = s.train_pct + s.val_pct + s.test_pct
        if total != 100:
            # Prefer adjusting train, then val, never let test hit 0.
            others = [x for x in ("train_pct", "val_pct", "test_pct") if x != which]
            diff = total - 100
            for o in others:
                cur = getattr(s, o)
                lo = 5
                new = max(lo, cur - diff)
                diff -= (cur - new)
                setattr(s, o, new)
                if diff == 0:
                    break
            # Final clamp if still off (rare)
            s.train_pct = max(5, 100 - s.val_pct - s.test_pct)

    # ---------------- model / training ----------------
    def compile_model(self):
        tf.keras.backend.clear_session()
        inp = tf.keras.Input(shape=(self.ds.n_features,))
        x, outs = inp, []
        for h in self.hidden:
            x = tf.keras.layers.Dense(h["units"], activation=h["act"])(x)
            outs.append(x)
        act, units, loss = self.out
        x = tf.keras.layers.Dense(units, activation=act)(x)
        outs.append(x)
        model = tf.keras.Model(inp, x)
        metrics = ["mae"] if self.ds.task == "regression" else ["accuracy"]
        model.compile(tf.keras.optimizers.Adam(0.01), loss=loss, metrics=metrics)
        probe = tf.keras.Model(inp, outs)
        return model, probe

    def start_training(self):
        errs = self.split.validate(self.ds.task)
        if errs:
            self.error = "; ".join(errs)
            return False
        try:
            p = self.ds.prepare(self.split)
        except Exception as e:
            self.error = f"Split failed: {e}"
            return False
        self.error = ""
        self.p = p
        self.train_state = {
            "phase": "fold" if self.split.mode == "kfold" else "holdout",
            "fold_i": 0,
            "epoch": 0,
            "paused": False,
            "sample": 0,
            "train_losses": [], "val_losses": [], "test_losses": [],
            "train_metrics": [], "val_metrics": [], "test_metrics": [],
            "best_val": float("inf"), "best_epoch": 0,
            "overfit": False, "rise_count": 0,
            "fold_val_errors": [],
            "cv_mean": None, "cv_std": None,
            "final_train": None, "final_val": None, "final_test": None,
            "rng": np.random.default_rng(self.split.seed),
        }
        self._load_fold_arrays()
        self.model, self.probe = self.compile_model()
        return True

    def _load_fold_arrays(self):
        """Set X_tr/y_tr/X_va/y_va for the current phase (hold-out, fold, or final retrain)."""
        p, st = self.p, self.train_state
        if st["phase"] == "holdout":
            self.X_tr, self.y_tr = p.X_tr, p.y_tr
            self.X_va, self.y_va = p.X_va, p.y_va
            self.X_viz_raw = p.X_va_raw
        elif st["phase"] == "fold":
            tr_idx, va_idx = p.folds[st["fold_i"]]
            self.X_tr, self.y_tr = p.X_pool[tr_idx], p.y_pool[tr_idx]
            self.X_va, self.y_va = p.X_pool[va_idx], p.y_pool[va_idx]
            self.X_viz_raw = p.X_pool_raw[va_idx]
        else:  # final retrain on all pool
            self.X_tr, self.y_tr = p.X_pool, p.y_pool
            self.X_va, self.y_va = p.X_pool[:max(1, len(p.X_pool) // 10)], p.y_pool[:max(1, len(p.X_pool) // 10)]
            self.X_viz_raw = p.X_pool_raw[:len(self.X_va)]
        self.X_te, self.y_te = p.X_te, p.y_te

    def predict_classes(self, pred):
        return (pred[:, 0] > 0.5).astype(int) if pred.shape[1] == 1 else np.argmax(pred, axis=1)

    def _eval(self, X, y):
        """Return (loss, display_metric). Metric is MAE (original units) or accuracy."""
        out = self.model.evaluate(X, y, verbose=0)
        loss, met = float(out[0]), float(out[1])
        if self.ds.task == "regression":
            met *= self.p.y_std
        return loss, met

    def _display_error(self, loss, met):
        """Scalar 'error' used for curves / overfitting: loss for regression, (1 - accuracy) for classification."""
        if self.ds.task == "regression":
            return loss
        return 1.0 - met

    def train_step(self):
        st = self.train_state
        if st is None or st["paused"] or st["phase"] == "done":
            return
        max_ep = self.args.epochs
        if st["phase"] == "final":
            max_ep = max(20, self.args.epochs // 2)

        if st["epoch"] >= max_ep:
            self._finish_phase()
            return

        idx = st["rng"].permutation(len(self.X_tr))
        bs = max(8, len(idx) // 15)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            self.model.train_on_batch(self.X_tr[b], self.y_tr[b])

        tr_l, tr_m = self._eval(self.X_tr, self.y_tr)
        va_l, va_m = self._eval(self.X_va, self.y_va)
        st["epoch"] += 1
        st["train_losses"].append(tr_l); st["val_losses"].append(va_l)
        st["train_metrics"].append(tr_m); st["val_metrics"].append(va_m)
        # Test curve only if didactic peeking is enabled (never used for model selection).
        if self.split.show_test_curve:
            te_l, te_m = self._eval(self.X_te, self.y_te)
            st["test_losses"].append(te_l); st["test_metrics"].append(te_m)

        # Best epoch by validation loss (model selection — test never used)
        if va_l < st["best_val"] - 1e-9:
            st["best_val"] = va_l
            st["best_epoch"] = st["epoch"]

        # Overfitting heuristic: val error rises while train falls for several epochs
        if len(st["val_losses"]) >= 4:
            te = [self._display_error(st["train_losses"][i], st["train_metrics"][i])
                  for i in range(-4, 0)]
            ve = [self._display_error(st["val_losses"][i], st["val_metrics"][i])
                  for i in range(-4, 0)]
            train_falling = te[-1] < te[0] - 1e-5
            val_rising = ve[-1] > ve[0] + 1e-5 and ve[-1] > ve[-2]
            if train_falling and val_rising:
                st["rise_count"] += 1
            else:
                st["rise_count"] = max(0, st["rise_count"] - 1)
            st["overfit"] = st["rise_count"] >= 3 and st["epoch"] > st["best_epoch"] + 2

    def _finish_phase(self):
        st = self.train_state
        if st["phase"] == "holdout":
            st["final_train"] = self._display_error(st["train_losses"][-1], st["train_metrics"][-1])
            be = max(1, st["best_epoch"])
            st["final_val"] = self._display_error(st["val_losses"][be - 1], st["val_metrics"][be - 1])
            # Evaluate test exactly once on the final model (never during training by default).
            te_l, te_m = self._eval(self.X_te, self.y_te)
            st["final_test"] = self._display_error(te_l, te_m)
            st["final_test_loss"], st["final_test_metric"] = te_l, te_m
            st["phase"] = "done"
            self.step = 6
        elif st["phase"] == "fold":
            # Error estimate for this fold = final validation loss (or 1-acc)
            ve = self._display_error(st["val_losses"][-1], st["val_metrics"][-1])
            st["fold_val_errors"].append(ve)
            st["fold_i"] += 1
            if st["fold_i"] < len(self.p.folds):
                # Next fold: fresh model, clear curves for current fold view
                st["epoch"] = 0
                st["train_losses"].clear(); st["val_losses"].clear(); st["test_losses"].clear()
                st["train_metrics"].clear(); st["val_metrics"].clear(); st["test_metrics"].clear()
                st["best_val"] = float("inf"); st["best_epoch"] = 0
                st["overfit"] = False; st["rise_count"] = 0; st["sample"] = 0
                self._load_fold_arrays()
                self.model, self.probe = self.compile_model()
            else:
                errs = np.array(st["fold_val_errors"])
                st["cv_mean"], st["cv_std"] = float(errs.mean()), float(errs.std())
                # Retrain final model on all training-pool data; test reported once at the end
                st["phase"] = "final"
                st["epoch"] = 0
                st["train_losses"].clear(); st["val_losses"].clear(); st["test_losses"].clear()
                st["train_metrics"].clear(); st["val_metrics"].clear(); st["test_metrics"].clear()
                st["best_val"] = float("inf"); st["best_epoch"] = 0
                st["overfit"] = False; st["rise_count"] = 0
                self._load_fold_arrays()
                self.model, self.probe = self.compile_model()
        elif st["phase"] == "final":
            st["final_train"] = self._display_error(st["train_losses"][-1], st["train_metrics"][-1])
            st["final_val"] = st["cv_mean"]  # Error_Estimate from CV
            te_l, te_m = self._eval(self.X_te, self.y_te)  # once, after final retrain
            st["final_test"] = self._display_error(te_l, te_m)
            st["final_test_loss"], st["final_test_metric"] = te_l, te_m
            st["phase"] = "done"
            self.step = 6

    # ---------------- common drawing ----------------
    def header(self):
        ui = self.ui
        ui.text("NN Visualizer", (12, 14), font=ui.big)
        x = 200
        for i, s in enumerate(STEPS):
            col = ACCENT if i == self.step else (TEXT if i < self.step else DIM)
            r = ui.text(f"{i + 1}.{s}", (x, 18), col, ui.small)
            x = r.right + 14
        pygame.draw.line(self.screen, PANEL, (0, 50), (W, 50), 2)

    def nav(self, next_label="Next >", next_enabled=True):
        ui = self.ui
        if self.step > 0:
            ui.button((20, H - 60, 120, 40), "< Back", lambda m: self.go(self.step - 1))
        else:
            ui.button((20, H - 60, 120, 40), "< Menu", lambda m: self.to_menu())
        ui.button((W - 160, H - 60, 140, 40), next_label,
                  lambda m: self.go(self.step + 1), active=True, enabled=next_enabled)
        if self.error:
            ui.text(self.error, (160, H - 50), (240, 120, 120), ui.small)

    def go(self, step):
        step = max(0, min(6, step))
        if step == 5:
            if not self.start_training():
                return
        if step < 5:
            self.train_state = None
        self.step = step

    def mini_network(self, rect):
        ui = self.ui
        sizes = self.layer_sizes()
        names = ["in"] + [h["act"] for h in self.hidden] + [self.out[0]]
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        n = len(sizes)
        for i, (s, nm) in enumerate(zip(sizes, names)):
            x = rect.x + 40 + i * (rect.w - 80) / max(n - 1, 1)
            hgt = min(rect.h - 60, 8 + 6 * min(s, 40))
            r = pygame.Rect(0, 0, 24, hgt); r.center = (int(x), rect.centery)
            pygame.draw.rect(self.screen, ACCENT if i in (0, n - 1) else SEL, r, border_radius=6)
            ui.text(s, (x, r.top - 16), anchor="center")
            ui.text(nm, (x, r.bottom + 12), DIM, ui.small, anchor="center")

    def draw_split_bar(self, rect):
        """Colored bar like lecture slides: green train, orange val, red test (folds for K-fold)."""
        ui = self.ui
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=6)
        s = self.split
        if s.mode == "holdout":
            parts = [("Train", s.train_pct, COL_TRAIN),
                     ("Val", s.val_pct, COL_VAL),
                     ("Test", s.test_pct, COL_TEST)]
            x = rect.x + 8
            for name, pct, col in parts:
                w = int((rect.w - 16) * pct / 100)
                pygame.draw.rect(self.screen, col, (x, rect.y + 28, max(2, w - 2), rect.h - 48), border_radius=4)
                if w > 40:
                    ui.text(f"{name} {pct}%", (x + w / 2, rect.y + 28 + (rect.h - 48) / 2),
                            (10, 10, 10), ui.small, anchor="center")
                x += w
        else:
            # Held-out test on the right; remaining split into K fold stripes
            test_w = int((rect.w - 16) * s.test_pct_kfold / 100)
            pool_w = rect.w - 16 - test_w
            fold_w = pool_w / s.k
            for i in range(s.k):
                # Alternate: one fold is "val" (orange), rest train (green)
                # Show all folds as green with one highlighted orange to illustrate CV
                x = rect.x + 8 + i * fold_w
                col = COL_VAL if i == 0 else COL_TRAIN
                pygame.draw.rect(self.screen, col, (x, rect.y + 28, max(2, fold_w - 2), rect.h - 48), border_radius=3)
                if fold_w > 28:
                    ui.text(f"F{i + 1}", (x + fold_w / 2, rect.y + 28 + (rect.h - 48) / 2),
                            (10, 10, 10), ui.small, anchor="center")
            pygame.draw.rect(self.screen, COL_TEST,
                             (rect.x + 8 + pool_w, rect.y + 28, max(2, test_w - 2), rect.h - 48), border_radius=4)
            if test_w > 40:
                ui.text(f"Test {s.test_pct_kfold}%",
                        (rect.x + 8 + pool_w + test_w / 2, rect.y + 28 + (rect.h - 48) / 2),
                        (10, 10, 10), ui.small, anchor="center")
            ui.text(f"{s.k}-fold on train pool  |  orange = example val fold  |  red = held-out test (never trained on)",
                    (rect.x + 8, rect.y + 6), DIM, ui.small)
        # Legend
        lx = rect.x + 8
        for name, col in (("Train", COL_TRAIN), ("Validation", COL_VAL), ("Test", COL_TEST)):
            pygame.draw.rect(self.screen, col, (lx, rect.bottom - 18, 12, 12))
            ui.text(name, (lx + 16, rect.bottom - 18), DIM, ui.small)
            lx += 110

    # ---------------- wizard steps ----------------
    def draw_dataset(self):
        ui = self.ui
        ui.text("Step 1 — Choose a public dataset", (40, 70), font=ui.big)
        for i, name in enumerate(DATASETS):
            ui.button((40, 120 + i * 50, 260, 40), name,
                      lambda m, n=name: self.select_dataset(n), active=(name == self.ds_name))
        d = self.ds
        ui.text(d.name, (340, 120), font=ui.big)
        for i, line in enumerate(d.summary_lines()):
            ui.text(line, (340, 160 + i * 24))
        ui.text(f"→ Input layer: {d.n_features} neurons (one per feature)", (340, 240), ACCENT)
        cols = d.features[:6] + [d.target_col]
        df = d.frame[cols].head(8)
        y0, cw = 290, 125
        for j, c in enumerate(cols):
            ui.text(str(c)[:15], (340 + j * cw, y0), DIM, ui.small)
        for r_i, (_, row) in enumerate(df.iterrows()):
            for j, c in enumerate(cols):
                v = row[c]
                s = f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)
                if c == d.target_col and d.class_names:
                    s = d.class_names[int(v)]
                ui.text(s, (340 + j * cw, y0 + 22 + r_i * 22), TEXT, ui.small)
        if d.n_features > 6:
            ui.text(f"... and {d.n_features - 6} more feature columns", (340, y0 + 210), DIM, ui.small)
        self.nav()

    def draw_split(self):
        ui, s = self.ui, self.split
        ui.text("Step 2 — Data split (train / validation / test)", (40, 70), font=ui.big)
        ui.text("Test data is never used for training or model selection.", (40, 105), DIM)

        ui.button((40, 140, 220, 40), "Hold-out", lambda m: setattr(s, "mode", "holdout"),
                  active=s.mode == "holdout")
        ui.button((280, 140, 280, 40), "K-fold cross-validation", lambda m: setattr(s, "mode", "kfold"),
                  active=s.mode == "kfold")

        if s.mode == "holdout":
            ui.spinner(40, 210, "Train %", s.train_pct, lambda v: self.set_pct("train_pct", v), lo=5, hi=90)
            ui.spinner(40, 255, "Validation %", s.val_pct, lambda v: self.set_pct("val_pct", v), lo=5, hi=90)
            ui.spinner(40, 300, "Test %", s.test_pct, lambda v: self.set_pct("test_pct", v), lo=5, hi=90)
            total = s.train_pct + s.val_pct + s.test_pct
            ui.text(f"Sum = {total}% {'(OK)' if total == 100 else '(must be 100)'}",
                    (40, 350), ACCENT if total == 100 else (240, 120, 120))
        else:
            ui.spinner(40, 210, "K folds", s.k, lambda v: setattr(s, "k", v), lo=2, hi=10)
            ui.spinner(40, 255, "Held-out test %", s.test_pct_kfold,
                       lambda v: setattr(s, "test_pct_kfold", v), lo=5, hi=40)
            ui.text("Remaining samples form the CV pool; each fold is val once.", (40, 305), DIM, ui.small)

        can_strat = self.ds.task != "regression"
        ui.button((40, 390, 200, 36),
                  f"Stratified: {'ON' if s.stratified else 'OFF'}",
                  lambda m: setattr(s, "stratified", not s.stratified) if can_strat else None,
                  active=s.stratified, enabled=can_strat)
        ui.text("(classification only — preserves class ratios in each fold/split)",
                (250, 398), DIM, ui.small)

        ui.spinner(40, 440, "Random seed", s.seed, lambda v: setattr(s, "seed", v), lo=0, hi=9999)

        ui.button((520, 440, 420, 36),
                  f"Show test curve (didactic, peeking): {'ON' if s.show_test_curve else 'OFF'}",
                  lambda m: setattr(s, "show_test_curve", not s.show_test_curve),
                  active=s.show_test_curve)
        ui.text("Default OFF. Peeking at test during training is not proper practice — "
                "test is evaluated once at the end.",
                (520, 482), (240, 160, 100) if s.show_test_curve else DIM, ui.small)

        self.draw_split_bar(pygame.Rect(40, 520, W - 80, 130))
        ok = not self.split.validate(self.ds.task)
        self.nav(next_enabled=ok)

    def toggle_select(self, i, mods):
        if mods & (pygame.KMOD_CTRL | pygame.KMOD_META):
            self.selected ^= {i}
        elif mods & pygame.KMOD_SHIFT and self.selected:
            a = min(self.selected); self.selected = set(range(min(a, i), max(a, i) + 1))
        else:
            self.selected = {i}

    def apply_selected(self, key, fn):
        for i in self.selected:
            self.hidden[i][key] = fn(self.hidden[i][key])

    def draw_architecture(self):
        ui = self.ui
        ui.text("Step 3 — Architecture", (40, 70), font=ui.big)
        ui.text(f"Input layer: {self.ds.n_features} neurons (fixed by dataset)", (40, 120), DIM)
        ui.spinner(40, 160, "Hidden layers", self.n_hidden, self.set_n_hidden, lo=0, hi=8)
        sizes = sorted({o[1] for o in self.out_opts})
        ui.text("Output size", (40, 216))
        ui.cycle((210, 210, 130, 30), sizes, self.out[1], self.set_output_units)
        ui.text(f"(auto: {self.out_opts[0][1]}; allowed for {self.ds.task}: {sizes})", (360, 216), DIM)
        self.mini_network(pygame.Rect(40, 280, W - 80, 380))
        self.nav()

    def draw_hidden(self):
        ui = self.ui
        ui.text("Step 4 — Hidden layers", (40, 70), font=ui.big)
        ui.text("Click a layer to select; Ctrl-click toggles, Shift-click selects a range.",
                (40, 105), DIM)
        if not self.hidden:
            ui.text("No hidden layers (go back to add some).", (40, 150))
        for i, h in enumerate(self.hidden):
            y = 140 + i * 44
            ui.button((40, y, 460, 38), f"Hidden {i + 1}:  {h['units']} neurons,  {h['act']}",
                      lambda m, i=i: self.toggle_select(i, m), active=i in self.selected)
        if self.hidden:
            yb = 140 + len(self.hidden) * 44 + 10
            ui.button((40, yb, 140, 34), "Select all",
                      lambda m: setattr(self, "selected", set(range(len(self.hidden)))))
            ui.button((190, yb, 140, 34), "Select none", lambda m: setattr(self, "selected", set()))
            x0 = 560
            ui.text(f"Selected: {', '.join(str(i + 1) for i in sorted(self.selected)) or 'none'}", (x0, 140))
            first = self.hidden[min(self.selected)] if self.selected else self.hidden[0]
            ui.text("Neurons", (x0, 190))
            for k, d in enumerate([-10, -1, 1, 10]):
                ui.button((x0 + 120 + k * 60, 184, 54, 30), f"{d:+d}",
                          lambda m, d=d: self.apply_selected("units", lambda u: max(1, min(256, u + d))))
            ui.text("Activation", (x0, 240))
            ui.cycle((x0 + 120, 234, 234, 30), ACTIVATIONS, first["act"],
                     lambda a: self.apply_selected("act", lambda _: a))
            for k, a in enumerate(ACTIVATIONS):
                ui.button((x0 + (k % 4) * 120, 284 + (k // 4) * 40, 112, 32), a,
                          lambda m, a=a: self.apply_selected("act", lambda _: a),
                          active=a == first["act"])
        self.mini_network(pygame.Rect(40, 500, W - 80, 200))
        self.nav()

    def draw_output(self):
        ui, d = self.ui, self.ds
        ui.text("Step 5 — Output layer", (40, 70), font=ui.big)
        ui.text(f"Task: {d.task}" + (f" ({d.n_classes} classes)" if d.n_classes else ""), (40, 115))
        ui.text("Sensible output configurations (loss chosen automatically):", (40, 150), DIM)
        for i, (act, units, loss) in enumerate(self.out_opts):
            ui.button((40, 190 + i * 50, 640, 40),
                      f"{act}  x {units} unit(s)   →   loss: {loss}",
                      lambda m, i=i: setattr(self, "out_idx", i), active=i == self.out_idx)
        act, units, loss = self.out
        y = 220 + len(self.out_opts) * 50
        for line in (f"Output activation: {act}", f"Output neurons: {units}",
                     f"Loss: {loss}", f"Metric: {'MAE' if d.task == 'regression' else 'accuracy'}"):
            ui.text(line, (40, y), ACCENT); y += 26
        self.mini_network(pygame.Rect(40, 500, W - 80, 200))
        self.nav("Train >")

    # ---------------- training view ----------------
    def shown(self, n):
        if n <= MAX_DRAWN:
            return list(range(n))
        half = (MAX_DRAWN - 1) // 2
        return list(range(half)) + [None] + list(range(n - (MAX_DRAWN - 1 - half), n))

    def draw_curves(self, r, series, labels, colors, best_epoch=0, log_scale=False):
        """Error chart with grid, axis ticks/titles, end-value labels, optional log y-scale."""
        ui = self.ui
        pygame.draw.rect(self.screen, (30, 33, 45), r)
        pygame.draw.rect(self.screen, DIM, r, 1)
        title = "Train / Val error" + (" / Test (peeking)" if any(
            lab.lower().startswith("test") for lab in labels) else "")
        title += "  [log]" if log_scale else "  [linear]"
        ui.text(title + "   (L = toggle log scale)", (r.x + 8, r.y + 4), DIM, ui.small)

        pad_l, pad_r, pad_t, pad_b = 52, 70, 28, 36
        plot = pygame.Rect(r.x + pad_l, r.y + pad_t, r.w - pad_l - pad_r, r.h - pad_t - pad_b)
        usable = [s for s in series if len(s) >= 1]
        if not usable:
            ui.text("Waiting for epochs…", plot.center, DIM, anchor="center")
            return

        n = max(len(s) for s in series)
        allv = np.concatenate([np.asarray(s, dtype=float) for s in usable])
        allv = allv[np.isfinite(allv)]
        if allv.size == 0:
            return

        def _fwd(v):
            v = float(v)
            if log_scale:
                return np.log10(max(v, 1e-6))
            return v

        def _inv(u):
            return 10 ** u if log_scale else u

        lo_u, hi_u = _fwd(allv.min()), _fwd(allv.max())
        if hi_u <= lo_u:
            hi_u = lo_u + (1.0 if not log_scale else 0.5)
        pad = (hi_u - lo_u) * 0.08
        lo_u, hi_u = lo_u - pad, hi_u + pad
        # Error is non-negative; keep linear axis from flooring below 0.
        if not log_scale and allv.min() >= 0:
            lo_u = max(0.0, lo_u)

        def y_of(v):
            u = _fwd(v)
            return plot.bottom - (u - lo_u) / (hi_u - lo_u) * plot.h

        def x_of(i):
            return plot.x + (i / max(n - 1, 1)) * plot.w

        # --- grid + y ticks ---
        n_yt = 5
        for ti in range(n_yt + 1):
            u = lo_u + (hi_u - lo_u) * ti / n_yt
            yy = plot.bottom - ti / n_yt * plot.h
            pygame.draw.line(self.screen, (48, 52, 68), (plot.x, yy), (plot.right, yy), 1)
            val = _inv(u)
            lbl = f"{val:.0e}" if log_scale or val < 0.01 else (f"{val:.3f}" if val < 1 else f"{val:.2f}")
            ui.text(lbl, (plot.x - 6, yy), DIM, ui.small, anchor="midright")

        # --- x ticks (epochs) ---
        n_xt = min(6, max(2, n))
        for ti in range(n_xt):
            ep = 1 + int(round(ti * (n - 1) / max(n_xt - 1, 1)))
            xx = x_of(ep - 1)
            pygame.draw.line(self.screen, (48, 52, 68), (xx, plot.y), (xx, plot.bottom), 1)
            ui.text(str(ep), (xx, plot.bottom + 4), DIM, ui.small, anchor="midtop")

        pygame.draw.rect(self.screen, DIM, plot, 1)
        ui.text("epoch", (plot.centerx, r.bottom - 14), DIM, ui.small, anchor="center")
        # Rotated-ish y title (drawn vertically as stacked letters is ugly; use short label)
        ui.text("error", (r.x + 6, plot.centery), DIM, ui.small, anchor="center")

        # --- series ---
        for s, col, lab in zip(series, colors, labels):
            if len(s) < 1:
                continue
            pts = [(x_of(i), y_of(v)) for i, v in enumerate(s)]
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, col, False, pts, 2)
            else:
                pygame.draw.circle(self.screen, col, (int(pts[0][0]), int(pts[0][1])), 3)
            # End-value label
            lx, ly = pts[-1]
            ui.text(f"{s[-1]:.3g}", (lx + 6, ly), col, ui.small, anchor="midleft")
            pygame.draw.circle(self.screen, col, (int(lx), int(ly)), 3)

        if best_epoch > 0 and n > 1:
            bx = x_of(best_epoch - 1)
            pygame.draw.line(self.screen, ACCENT, (bx, plot.y), (bx, plot.bottom), 1)
            ui.text("best val", (bx + 3, plot.y + 2), ACCENT, ui.small)

        # Legend + last-value readout
        lx = plot.x
        readout = []
        for lab, col, s in zip(labels, colors, series):
            pygame.draw.line(self.screen, col, (lx, r.y + 16), (lx + 14, r.y + 16), 2)
            ui.text(lab, (lx + 18, r.y + 10), DIM, ui.small)
            if s:
                readout.append(f"{lab}={s[-1]:.4g}")
            lx += 70
        if readout:
            ui.text("  |  ".join(readout), (lx + 10, r.y + 10), TEXT, ui.small)

    def draw_train(self):
        ui, d, st, p = self.ui, self.ds, self.train_state, self.p
        # Network viz uses a validation sample
        n_viz = len(self.X_va)
        st["sample"] %= max(1, n_viz)
        x = self.X_va[st["sample"]:st["sample"] + 1]
        acts = [x[0]] + [np.asarray(a)[0] for a in self.probe(x, training=False)]
        weights = [l.get_weights()[0] for l in self.model.layers if l.get_weights()]
        sizes = self.layer_sizes()
        nl = len(sizes)
        left, right, top, bottom = 230, W - 360, 70, 420
        shown = [self.shown(n) for n in sizes]
        pos = []
        for i, idx in enumerate(shown):
            lx = left + i * (right - left) / (nl - 1)
            gap = (bottom - top) / len(idx)
            pos.append([(int(lx), int(top + gap * (j + 0.5))) for j in range(len(idx))])
        rad = int(max(6, min(16, (bottom - top) / max(len(s) for s in shown) / 2 - 3)))

        for li, w in enumerate(weights):
            m = np.max(np.abs(w)) + 1e-8
            for a, ia in enumerate(shown[li]):
                for b, ib in enumerate(shown[li + 1]):
                    if ia is None or ib is None:
                        continue
                    v = w[ia, ib] / m
                    k = abs(v)
                    base = COL_TRAIN if v > 0 else COL_TEST
                    col = tuple(int(BG[c] + (base[c] - BG[c]) * (0.12 + 0.88 * k)) for c in range(3))
                    pygame.draw.line(self.screen, col, pos[li][a], pos[li + 1][b], max(1, int(1 + 3 * k)))

        names = ["Input"] + [f"H{i + 1}({h['act']})" for i, h in enumerate(self.hidden)] + [f"Out({self.out[0]})"]
        for li, a in enumerate(acts):
            a = np.asarray(a, dtype=np.float32)
            if li == 0:
                norm = np.clip((a + 2.5) / 5.0, 0, 1)
            elif li == nl - 1 and self.out[0] != "linear":
                norm = a
            else:
                rng = np.max(np.abs(a)) + 1e-8
                norm = np.clip(np.abs(a) / rng, 0, 1)
            for j, (pp, idx) in enumerate(zip(pos[li], shown[li])):
                if idx is None:
                    ui.text(f"...{sizes[li] - MAX_DRAWN + 1}", pp, DIM, ui.small, anchor="center")
                    continue
                g = float(norm[idx])
                col = (int(30 + 225 * g), int(40 + 180 * g), int(70 + 60 * g))
                pygame.draw.circle(self.screen, col, pp, rad)
                pygame.draw.circle(self.screen, (200, 200, 220), pp, rad, 1)
            ui.text(f"{names[li]}[{sizes[li]}]", (pos[li][0][0], 55), DIM, ui.small, anchor="center")

        raw = self.X_viz_raw[st["sample"]]
        for pp, idx in zip(pos[0], shown[0]):
            if idx is not None:
                ui.text(f"{str(d.features[idx])[:18]}={raw[idx]:.3g}",
                        (pp[0] - rad - 6, pp[1]), TEXT, ui.small, anchor="midright")

        out = acts[-1]
        if d.task == "regression":
            pv = out[0] * p.y_std + p.y_mean
            tv = self.y_va[st["sample"]] * p.y_std + p.y_mean
            ui.text(f"pred {pv:.3g}  true {tv:.3g}", (pos[-1][0][0] + rad + 8, pos[-1][0][1]), TEXT, ui.small)
        else:
            pred = int(self.predict_classes(out[None])[0])
            true = int(self.y_va[st["sample"]])
            if len(out) == 1:
                ui.text(f"P({d.class_names[1]})={out[0]*100:.0f}% pred={d.class_names[pred]}",
                        (pos[-1][0][0] + rad + 8, pos[-1][0][1]),
                        (120, 230, 120) if pred == true else (240, 120, 120), ui.small)
            else:
                for pp, idx in zip(pos[-1], shown[-1]):
                    if idx is None:
                        continue
                    tag = ("←p" if idx == pred else "") + ("[t]" if idx == true else "")
                    col = (120, 230, 120) if idx == pred == true else (240, 120, 120) if idx == pred else TEXT
                    ui.text(f"{d.class_names[idx][:10]} {out[idx]*100:.0f}%{tag}",
                            (pp[0] + rad + 6, pp[1]), col, ui.small, anchor="midleft")

        # Status panel
        if st["phase"] == "fold":
            phase = f"Fold {st['fold_i'] + 1}/{len(p.folds)}"
        elif st["phase"] == "final":
            phase = "Final retrain (all train data)"
        elif st["phase"] == "done":
            phase = "DONE"
        else:
            phase = "Hold-out"
        status = "PAUSED" if st["paused"] else phase
        peek = self.split.show_test_curve
        if d.task == "regression":
            mtxt = (f"Tr MAE {st['train_metrics'][-1]:.3g}  Val MAE {st['val_metrics'][-1]:.3g}"
                    if st["train_metrics"] else "")
            if peek and st["test_metrics"]:
                mtxt += f"  Test MAE {st['test_metrics'][-1]:.3g} (peeking)"
        else:
            mtxt = (f"Tr acc {st['train_metrics'][-1]*100:.1f}%  Val {st['val_metrics'][-1]*100:.1f}%"
                    if st["train_metrics"] else "")
            if peek and st["test_metrics"]:
                mtxt += f"  Test {st['test_metrics'][-1]*100:.1f}% (peeking)"
        gap = ""
        if st["train_losses"]:
            te = self._display_error(st["train_losses"][-1], st["train_metrics"][-1])
            ve = self._display_error(st["val_losses"][-1], st["val_metrics"][-1])
            gap = f"  gap(train−val error)={te - ve:+.4f}"
        loss_line = (f"{d.name} | {status} | Ep {st['epoch']}/{self.args.epochs}  "
                     f"loss Tr {st['train_losses'][-1] if st['train_losses'] else 0:.3f} / "
                     f"Val {st['val_losses'][-1] if st['val_losses'] else 0:.3f}")
        if peek and st["test_losses"]:
            loss_line += f" / Te {st['test_losses'][-1]:.3f} (peek)"
        else:
            loss_line += "  |  test held out until Results"
        ui.text(loss_line, (20, 430), font=self.med)
        ui.text(mtxt + gap, (20, 455), TEXT, ui.small)
        ui.text(f"Best val @ epoch {st['best_epoch']}" + ("   ⚠ Overfitting?" if st["overfit"] else ""),
                (20, 475), (240, 180, 80) if st["overfit"] else DIM)

        # Fold table (K-fold)
        if self.split.mode == "kfold":
            tx, ty = W - 340, 70
            ui.text("Per-fold val error", (tx, ty), DIM, ui.small)
            for i, e in enumerate(st["fold_val_errors"]):
                ui.text(f"Fold {i + 1}: {e:.4f}", (tx, ty + 18 + i * 16), TEXT, ui.small)
            if st["cv_mean"] is not None:
                ui.text(f"Error_Estimate = {st['cv_mean']:.4f} ± {st['cv_std']:.4f}",
                        (tx, ty + 18 + len(st['fold_val_errors']) * 16 + 4), ACCENT, ui.small)

        # Error curves: train + val always; test only if didactic peeking is on.
        err_tr = [self._display_error(a, b) for a, b in zip(st["train_losses"], st["train_metrics"])]
        err_va = [self._display_error(a, b) for a, b in zip(st["val_losses"], st["val_metrics"])]
        series, labels, colors = [err_tr, err_va], ["Train", "Val"], [COL_TRAIN, COL_VAL]
        if self.split.show_test_curve and st["test_losses"]:
            err_te = [self._display_error(a, b) for a, b in zip(st["test_losses"], st["test_metrics"])]
            series.append(err_te); labels.append("Test*"); colors.append(COL_TEST)
        self.draw_curves(pygame.Rect(20, 500, W - 40, 175),
                         series, labels, colors, st["best_epoch"], log_scale=self.log_scale)

        ui.button((20, H - 55, 120, 40), "< Back", lambda m: self.go(4))
        ui.text("SPACE pause  N next sample  R reset  L log scale  B back  ESC quit",
                (160, H - 40), DIM, ui.small)

    def draw_results(self):
        ui, st, d = self.ui, self.train_state, self.ds
        ui.text("Results — train / validation / test comparison", (40, 70), font=ui.big)
        if st is None or st["final_test"] is None:
            ui.text("Finish training to see results.", (40, 120))
            ui.button((20, H - 60, 120, 40), "< Back", lambda m: self.go(4))
            return

        kind = "error (1 − accuracy)" if d.task != "regression" else "error (loss / MSE)"
        ui.text(f"Lower is better — {kind}. Test evaluated once on the final model "
                "(never used for training or model selection).",
                (40, 110), DIM)

        vals = [("Train", st["final_train"], COL_TRAIN),
                ("Val / CV estimate", st["final_val"], COL_VAL),
                ("Test", st["final_test"], COL_TEST)]
        # Bar chart
        chart = pygame.Rect(80, 180, W - 160, 360)
        pygame.draw.rect(self.screen, PANEL, chart, border_radius=8)
        mx = max(v for _, v, _ in vals) + 1e-8
        bw = 140
        gap = (chart.w - 3 * bw) / 4
        for i, (name, v, col) in enumerate(vals):
            x = chart.x + gap + i * (bw + gap)
            hgt = int((chart.h - 80) * (v / mx))
            bar = pygame.Rect(x, chart.bottom - 50 - hgt, bw, max(2, hgt))
            pygame.draw.rect(self.screen, col, bar, border_radius=6)
            ui.text(f"{v:.4f}", (x + bw / 2, bar.top - 22), TEXT, anchor="center")
            ui.text(name, (x + bw / 2, chart.bottom - 30), DIM, anchor="center")

        y = 560
        if self.split.mode == "kfold" and st["cv_mean"] is not None:
            ui.text(f"K-fold Error_Estimate (mean of val errors) = {st['cv_mean']:.4f} ± {st['cv_std']:.4f}",
                    (40, y), ACCENT)
            ui.text("Fold errors: " + ", ".join(f"{e:.4f}" for e in st["fold_val_errors"]),
                    (40, y + 24), DIM, ui.small)
            y += 50
        gap = st["final_train"] - st["final_val"]
        ui.text(f"Train − Val gap: {gap:+.4f}" +
                ("  → possible overfitting if train << val" if gap < -0.05 else ""),
                (40, y), (240, 180, 80) if gap < -0.05 else TEXT)
        ui.button((20, H - 60, 160, 40), "< Reconfigure", lambda m: self.go(0))
        ui.button((W - 200, H - 60, 180, 40), "Train again", lambda m: self.go(5), active=True)

    # ---------------- main loop ----------------
    def draw(self):
        self.screen.fill(BG)
        self.ui.begin()
        self.header()
        drawers = [self.draw_dataset, self.draw_split, self.draw_architecture,
                   self.draw_hidden, self.draw_output, self.draw_train, self.draw_results]
        drawers[self.step]()
        pygame.display.flip()

    def handle(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
            mods = pygame.key.get_mods() | (pygame.KMOD_SHIFT if e.button == 3 else 0)
            self.ui.click(e.pos, mods)
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                return False
            if self.step == 5 and self.train_state:
                st = self.train_state
                if e.key == pygame.K_SPACE: st["paused"] = not st["paused"]
                elif e.key == pygame.K_n: st["sample"] = (st["sample"] + 1) % max(1, len(self.X_va))
                elif e.key == pygame.K_r:
                    self.start_training()
                elif e.key == pygame.K_l:
                    self.log_scale = not self.log_scale
                elif e.key == pygame.K_b: self.go(4)
            elif e.key == pygame.K_RETURN and self.step < 5:
                self.go(self.step + 1)
        return True

    def to_menu(self):
        self.result = "menu"
        self.running = False

    def run(self):
        script = AutoScript(self) if self.args.auto else None
        self.result = "quit"
        self.running = True
        running = True
        while running and self.running:
            for e in pygame.event.get():
                running = self.handle(e) and running
            if self.step == 5 and self.train_state and self.train_state["phase"] != "done":
                self.train_step()
                st = self.train_state
                if not st["paused"] and st["epoch"] and st["epoch"] % 10 == 0:
                    st["sample"] = (st["sample"] + 1) % max(1, len(self.X_va))
            self.draw()
            if script and not script.tick():
                running = False
            self.clock.tick(30)
        return self.result


class AutoScript:
    """Headless demo: walk the wizard, train, capture screenshots, exit."""

    def __init__(self, app):
        self.app = app
        self.frame = 0
        self.out = app.args.screenshot_dir
        os.makedirs(self.out, exist_ok=True)
        self.train_frames = max(30, int(app.args.seconds * 30))
        self.phase_shot = set()
        self.start_train_frame = None
        self.done = False

    def shot(self, name):
        path = os.path.join(self.out, name)
        pygame.image.save(self.app.screen, path)
        print("saved", path, flush=True)

    def tick(self):
        a, f = self.app, self.frame
        self.frame += 1
        # Wizard steps 0..4
        if a.step < 5:
            if f % 4 == 3:
                if a.step == 1:  # split step — capture before leaving
                    tag = "kfold" if a.split.mode == "kfold" else "holdout"
                    self.shot(f"2_split_{tag}.png")
                elif a.step == 0:
                    self.shot("1_dataset.png")
                elif a.step == 2:
                    self.shot("3_architecture.png")
                elif a.step == 3:
                    a.selected = set(range(len(a.hidden)))
                    a.apply_selected("act", lambda _: "relu")
                    self.shot("4_hidden.png")
                elif a.step == 4:
                    self.shot("5_output.png")
                a.go(a.step + 1)
                if a.step == 5:
                    self.start_train_frame = self.frame
            return True

        # Training
        st = a.train_state
        if st is None:
            return False
        tag = a.ds_name.lower().replace(" ", "_")
        mode = a.split.mode

        if st["phase"] == "fold" and st["fold_i"] == 0 and st["epoch"] == 5 and "kfold_train" not in self.phase_shot:
            self.shot(f"6_train_kfold_{tag}.png")
            self.phase_shot.add("kfold_train")
        if st["phase"] == "holdout" and st["epoch"] == 8 and "holdout_train" not in self.phase_shot:
            self.shot(f"6_train_holdout_{tag}.png")
            self.phase_shot.add("holdout_train")

        # Force finish if we've spent enough wall frames OR enough epochs overall
        elapsed = self.frame - (self.start_train_frame or self.frame)
        # Speed up auto: jump phases after enough epochs per fold
        if mode == "kfold" and st["phase"] == "fold" and st["epoch"] >= max(8, a.args.epochs // 10):
            # Manually advance by setting epoch high
            st["epoch"] = a.args.epochs
        if mode == "kfold" and st["phase"] == "final" and st["epoch"] >= max(8, a.args.epochs // 10):
            st["epoch"] = a.args.epochs
        if mode == "holdout" and st["phase"] == "holdout" and st["epoch"] >= max(15, a.args.epochs // 5):
            st["epoch"] = a.args.epochs

        if a.step == 6 and not self.done:
            self.shot(f"7_results_{mode}_{tag}.png")
            tr = st["final_train"]; va = st["final_val"]; te = st["final_test"]
            print(f"RESULT mode={mode} dataset={a.ds_name} train_err={tr:.4f} "
                  f"val_or_cv={va:.4f} test_err={te:.4f}"
                  + (f" cv_std={st['cv_std']:.4f}" if st.get("cv_std") is not None else ""),
                  flush=True)
            self.done = True
            return False
        if elapsed > self.train_frames * 8 and a.step == 5:
            # Safety timeout: force done
            print("auto timeout — forcing results", flush=True)
            if st["train_losses"]:
                st["final_train"] = a._display_error(st["train_losses"][-1], st["train_metrics"][-1])
                st["final_val"] = a._display_error(st["val_losses"][-1], st["val_metrics"][-1])
                te_l, te_m = a._eval(a.X_te, a.y_te)
                st["final_test"] = a._display_error(te_l, te_m)
                if mode == "kfold" and st["cv_mean"] is None and st["fold_val_errors"]:
                    errs = np.array(st["fold_val_errors"])
                    st["cv_mean"], st["cv_std"] = float(errs.mean()), float(errs.std())
                    st["final_val"] = st["cv_mean"]
            else:
                st["final_train"] = st["final_val"] = st["final_test"] = 0.0
            st["phase"] = "done"
            a.step = 6
        return True


def run_menu(args):
    """Start screen: choose Dataset Training or Live Evolution Training."""
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Neural Network Visualizer")
    ui, clock, choice = UI(screen), pygame.time.Clock(), [None]
    frame = 0
    while choice[0] is None:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return "quit"
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                ui.click(e.pos, 0)
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_1, pygame.K_d):
                choice[0] = "dataset"
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_2, pygame.K_e):
                choice[0] = "evolution"
        screen.fill(BG)
        ui.begin()
        ui.text("Neural Network Visualizer", (W // 2, 130), font=ui.big, anchor="center")
        ui.text("Choose a mode", (W // 2, 170), DIM, anchor="center")
        ui.button((W // 2 - 420, 240, 400, 220), "1  Dataset Training",
                  lambda m: choice.__setitem__(0, "dataset"), active=True)
        ui.button((W // 2 + 20, 240, 400, 220), "2  Live Evolution Training",
                  lambda m: choice.__setitem__(0, "evolution"), active=True)
        ui.text("Keras on sklearn datasets: train/val/test, K-fold,", (W // 2 - 220, 490), DIM, ui.small, anchor="center")
        ui.text("live activations and error curves.", (W // 2 - 220, 510), DIM, ui.small, anchor="center")
        ui.text("Cars with NN brains learn to drive a track", (W // 2 + 220, 490), DIM, ui.small, anchor="center")
        ui.text("via a genetic algorithm (sensors -> throttle/steer).", (W // 2 + 220, 510), DIM, ui.small, anchor="center")
        ui.text("ESC quit", (W // 2, H - 40), DIM, ui.small, anchor="center")
        pygame.display.flip()
        frame += 1
        if args.auto and frame == 3:
            os.makedirs(args.screenshot_dir, exist_ok=True)
            path = os.path.join(args.screenshot_dir, "0_menu.png")
            pygame.image.save(screen, path)
            print("saved", path, flush=True)
            return args.auto_target
        clock.tick(30)
    return choice[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["menu", "dataset", "evolution"], default="menu")
    ap.add_argument("--auto-target", choices=["dataset", "evolution"], default="dataset",
                    help="In --auto with --mode menu: which mode to enter after the menu screenshot")
    ap.add_argument("--dataset", default="Iris", choices=list(DATASETS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--screenshot-dir", default="screenshots")
    ap.add_argument("--split", choices=["holdout", "kfold"], default="holdout")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--train-pct", type=int, default=70)
    ap.add_argument("--val-pct", type=int, default=15)
    ap.add_argument("--test-pct", type=int, default=15)
    ap.add_argument("--test-pct-kfold", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stratified", dest="stratified", action="store_true", default=True)
    g.add_argument("--no-stratified", dest="stratified", action="store_false")
    ap.add_argument("--show-test-curve", action="store_true", default=False,
                    help="Didactic: plot test during training (default off)")
    # Evolution mode
    ap.add_argument("--track", default="Oval", choices=["Oval", "Curvy", "Kidney"])
    ap.add_argument("--population", type=int, default=50)
    ap.add_argument("--gen-time", type=int, default=20, help="max seconds per generation")
    ap.add_argument("--generations", type=int, default=8, help="generations to run in --auto evolution")
    args = ap.parse_args()

    pygame.init()
    mode = args.mode
    while mode not in ("quit", None):
        if mode == "menu":
            mode = run_menu(args)
            continue
        if mode == "dataset":
            res = App(args).run()
        else:
            from evolution.app import EvoApp
            res = EvoApp(args).run()
        mode = "menu" if (res == "menu" and not args.auto) else "quit"
    pygame.quit()


if __name__ == "__main__":
    main()
    sys.exit(0)
