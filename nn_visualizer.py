"""Neural Network Visualizer: a Pygame wizard to configure a tf.keras network on a public
dataset, then train it while visualizing activations and weights of every layer live."""
import os
import sys
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pygame
import tensorflow as tf

from datasets import DATASETS, load_dataset
from widgets import UI, BG, PANEL, TEXT, DIM, ACCENT, SEL

W, H = 1280, 780
ACTIVATIONS = ["relu", "tanh", "sigmoid", "elu", "leaky_relu", "selu", "linear"]
STEPS = ["Dataset", "Architecture", "Hidden layers", "Output", "Train"]
MAX_DRAWN = 16  # neurons drawn per layer before collapsing with an ellipsis


def output_options(task, n_classes):
    """Sensible (activation, units, loss) combos for the task."""
    if task == "regression":
        return [("linear", 1, "mse")]
    if task == "binary":
        return [("sigmoid", 1, "binary_crossentropy"), ("softmax", 2, "sparse_categorical_crossentropy")]
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
        self.select_dataset(args.dataset)
        self.n_hidden = 2
        self.hidden = [{"units": 8, "act": "relu"}, {"units": 6, "act": "relu"}]
        self.selected = {0}
        self.model = None

    # ---------------- configuration ----------------
    def select_dataset(self, name):
        try:
            if name not in self.ds_cache:
                self.ds_cache[name] = load_dataset(name)
            self.ds = self.ds_cache[name]
            self.ds_name = name
            self.error = ""
        except Exception as e:  # e.g. no network for California Housing
            self.error = f"Could not load {name}: {e}"
            if not hasattr(self, "ds"):
                self.select_dataset("Iris")
            return
        self.out_opts = output_options(self.ds.task, self.ds.n_classes)
        self.out_idx = 0

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
        # Keep output size consistent with the activation/loss: pick the option matching the size.
        for i, o in enumerate(self.out_opts):
            if o[1] == units:
                self.out_idx = i

    def layer_sizes(self):
        return [self.ds.n_features] + [h["units"] for h in self.hidden] + [self.out[1]]

    # ---------------- model / training ----------------
    def build(self):
        tf.keras.backend.clear_session()
        self.ds.prepare()
        inp = tf.keras.Input(shape=(self.ds.n_features,))
        x, outs = inp, []
        for h in self.hidden:
            x = tf.keras.layers.Dense(h["units"], activation=h["act"])(x)
            outs.append(x)
        act, units, loss = self.out
        x = tf.keras.layers.Dense(units, activation=act)(x)
        outs.append(x)
        self.model = tf.keras.Model(inp, x)
        metrics = ["mae"] if self.ds.task == "regression" else ["accuracy"]
        self.model.compile(tf.keras.optimizers.Adam(0.01), loss=loss, metrics=metrics)
        self.probe = tf.keras.Model(inp, outs)
        self.epoch, self.loss, self.metric, self.val_metric = 0, 0.0, 0.0, 0.0
        self.losses, self.sample, self.paused = [], 0, False
        self.rng = np.random.default_rng()

    def predict_classes(self, p):
        return (p[:, 0] > 0.5).astype(int) if p.shape[1] == 1 else np.argmax(p, axis=1)

    def train_step(self):
        """One epoch of mini-batch training per frame (keeps the GUI responsive)."""
        if self.paused or self.epoch >= self.args.epochs:
            return
        d = self.ds
        idx = self.rng.permutation(len(d.X_tr))
        bs = max(16, len(idx) // 20)
        ls, ms = [], []
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            r = self.model.train_on_batch(d.X_tr[b], d.y_tr[b])
            ls.append(float(r[0])); ms.append(float(r[1]))
        self.epoch += 1
        self.loss, self.metric = float(np.mean(ls)), float(np.mean(ms))
        self.losses.append(self.loss)
        p = self.model(d.X_te, training=False).numpy()
        if d.task == "regression":
            self.metric *= d.y_std  # MAE in original target units
            self.val_metric = float(np.mean(np.abs(p[:, 0] - d.y_te))) * d.y_std
        else:
            self.val_metric = float(np.mean(self.predict_classes(p) == d.y_te))

    # ---------------- common drawing ----------------
    def header(self):
        ui = self.ui
        ui.text("Neural Network Visualizer", (20, 12), font=ui.big)
        x = 380
        for i, s in enumerate(STEPS):
            col = ACCENT if i == self.step else (TEXT if i < self.step else DIM)
            r = ui.text(f"{i + 1}. {s}", (x, 18), col)
            x = r.right + 30
        pygame.draw.line(self.screen, PANEL, (0, 50), (W, 50), 2)

    def nav(self, next_label="Next >"):
        ui = self.ui
        if self.step > 0:
            ui.button((20, H - 60, 120, 40), "< Back", lambda m: self.go(self.step - 1))
        ui.button((W - 160, H - 60, 140, 40), next_label, lambda m: self.go(self.step + 1), active=True)
        if self.error:
            ui.text(self.error, (160, H - 50), (240, 120, 120))

    def go(self, step):
        step = max(0, min(4, step))
        if step == 4:
            try:
                self.build()
            except Exception as e:
                self.error = f"Model build failed: {e}"
                return
        self.step = step

    def mini_network(self, rect):
        """Small preview of the configured layer sizes."""
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

    # ---------------- wizard steps ----------------
    def draw_dataset(self):
        ui = self.ui
        ui.text("Step 1 - Choose a public dataset", (40, 70), font=ui.big)
        for i, name in enumerate(DATASETS):
            ui.button((40, 120 + i * 50, 260, 40), name, lambda m, n=name: self.select_dataset(n),
                      active=(name == self.ds_name))
        d = self.ds
        ui.text(f"{d.name}", (340, 120), font=ui.big)
        for i, line in enumerate(d.summary_lines()):
            ui.text(line, (340, 160 + i * 24))
        ui.text(f"-> Input layer: {d.n_features} neurons (one per feature)", (340, 240), ACCENT)
        # Preview table: first rows, first columns
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

    def draw_architecture(self):
        ui = self.ui
        ui.text("Step 2 - Architecture", (40, 70), font=ui.big)
        ui.text(f"Input layer: {self.ds.n_features} neurons (fixed by dataset)", (40, 120), DIM)
        ui.spinner(40, 160, "Hidden layers", self.n_hidden, self.set_n_hidden, lo=0, hi=8)
        sizes = sorted({o[1] for o in self.out_opts})
        ui.text("Output size", (40, 216))
        ui.cycle((210, 210, 130, 30), sizes, self.out[1], self.set_output_units)
        auto = self.out_opts[0][1]
        ui.text(f"(auto: {auto}; allowed for {self.ds.task}: {sizes})", (360, 216), DIM)
        self.mini_network(pygame.Rect(40, 280, W - 80, 380))
        self.nav()

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

    def draw_hidden(self):
        ui = self.ui
        ui.text("Step 3 - Hidden layers", (40, 70), font=ui.big)
        ui.text("Click a layer to select; Ctrl-click toggles, Shift-click selects a range. "
                "Edits apply to all selected layers.", (40, 105), DIM)
        if not self.hidden:
            ui.text("No hidden layers (go back to add some).", (40, 150))
        for i, h in enumerate(self.hidden):
            y = 140 + i * 44
            ui.button((40, y, 460, 38), f"Hidden {i + 1}:  {h['units']} neurons,  {h['act']}",
                      lambda m, i=i: self.toggle_select(i, m), active=i in self.selected)
        if self.hidden:
            ui.button((40, 140 + len(self.hidden) * 44 + 10, 140, 34), "Select all",
                      lambda m: setattr(self, "selected", set(range(len(self.hidden)))))
            ui.button((190, 140 + len(self.hidden) * 44 + 10, 140, 34), "Select none",
                      lambda m: setattr(self, "selected", set()))
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
                          lambda m, a=a: self.apply_selected("act", lambda _: a), active=a == first["act"])
        self.mini_network(pygame.Rect(40, 480, W - 80, 220))
        self.nav()

    def draw_output(self):
        ui = self.ui
        d = self.ds
        ui.text("Step 4 - Output layer", (40, 70), font=ui.big)
        ui.text(f"Task: {d.task}" + (f" ({d.n_classes} classes)" if d.n_classes else ""), (40, 115))
        ui.text("Sensible output configurations (loss is chosen automatically):", (40, 150), DIM)
        for i, (act, units, loss) in enumerate(self.out_opts):
            ui.button((40, 190 + i * 50, 640, 40), f"{act}  x {units} unit(s)   ->   loss: {loss}",
                      lambda m, i=i: setattr(self, "out_idx", i), active=i == self.out_idx)
        act, units, loss = self.out
        y = 220 + len(self.out_opts) * 50
        ui.text(f"Output activation: {act}", (40, y), ACCENT)
        ui.text(f"Output neurons: {units}", (40, y + 26), ACCENT)
        ui.text(f"Loss: {loss}", (40, y + 52), ACCENT)
        ui.text(f"Metric: {'MAE' if d.task == 'regression' else 'accuracy'}", (40, y + 78), ACCENT)
        self.mini_network(pygame.Rect(40, 480, W - 80, 220))
        self.nav("Train >")

    # ---------------- training view ----------------
    def shown(self, n):
        """Indices drawn for a layer of n neurons (None = ellipsis slot)."""
        if n <= MAX_DRAWN:
            return list(range(n))
        half = (MAX_DRAWN - 1) // 2
        return list(range(half)) + [None] + list(range(n - (MAX_DRAWN - 1 - half), n))

    def draw_train(self):
        ui, d = self.ui, self.ds
        x = d.X_te[self.sample:self.sample + 1]
        acts = [x[0]] + [np.asarray(a)[0] for a in self.probe(x, training=False)]
        weights = [l.get_weights()[0] for l in self.model.layers if l.get_weights()]
        sizes = self.layer_sizes()
        nl = len(sizes)
        left, right, top, bottom = 260, W - 330, 80, 590
        shown = [self.shown(n) for n in sizes]
        pos = []
        for i, idx in enumerate(shown):
            lx = left + i * (right - left) / (nl - 1)
            gap = (bottom - top) / len(idx)
            pos.append([(int(lx), int(top + gap * (j + 0.5))) for j in range(len(idx))])
        maxn = max(len(s) for s in shown)
        rad = int(max(6, min(18, (bottom - top) / maxn / 2 - 3)))

        for li, w in enumerate(weights):
            m = np.max(np.abs(w)) + 1e-8
            for a, ia in enumerate(shown[li]):
                for b, ib in enumerate(shown[li + 1]):
                    if ia is None or ib is None:
                        continue
                    v = w[ia, ib] / m
                    k = abs(v)
                    base = (60, 200, 90) if v > 0 else (220, 70, 70)
                    col = tuple(int(BG[c] + (base[c] - BG[c]) * (0.12 + 0.88 * k)) for c in range(3))
                    pygame.draw.line(self.screen, col, pos[li][a], pos[li + 1][b], max(1, int(1 + 4 * k)))

        names = ["Input"] + [f"Hidden {i + 1} ({h['act']})" for i, h in enumerate(self.hidden)] + [f"Output ({self.out[0]})"]
        for li, a in enumerate(acts):
            a = np.asarray(a, dtype=np.float32)
            if li == 0:
                norm = np.clip((a + 2.5) / 5.0, 0, 1)
            elif li == nl - 1 and self.out[0] != "linear":
                norm = a
            else:
                rng = np.max(np.abs(a)) + 1e-8
                norm = np.clip(np.abs(a) / rng, 0, 1)
            for j, (p, idx) in enumerate(zip(pos[li], shown[li])):
                if idx is None:
                    ui.text(f"... {sizes[li] - MAX_DRAWN + 1} more", p, DIM, ui.small, anchor="center")
                    continue
                g = float(norm[idx])
                col = (int(30 + 225 * g), int(40 + 180 * g), int(70 + 60 * g))
                pygame.draw.circle(self.screen, col, p, rad)
                pygame.draw.circle(self.screen, (200, 200, 220), p, rad, 1)
                if rad >= 14:
                    ui.text(f"{a[idx]:.2f}", p, (0, 0, 0) if g > 0.6 else TEXT, ui.small, anchor="center")
            ui.text(f"{names[li]} [{sizes[li]}]", (pos[li][0][0], 62), DIM, ui.small, anchor="center")

        # Input labels
        raw = d.X_te_raw[self.sample]
        for p, idx in zip(pos[0], shown[0]):
            if idx is not None:
                ui.text(f"{str(d.features[idx])[:22]} = {raw[idx]:.3g}", (p[0] - rad - 8, p[1]), TEXT,
                        ui.small, anchor="midright")

        # Output labels and verdict
        out = acts[-1]
        if d.task == "regression":
            pv = out[0] * d.y_std + d.y_mean
            tv = d.y_te[self.sample] * d.y_std + d.y_mean
            ui.text(f"pred {pv:.3g}", (pos[-1][0][0] + rad + 10, pos[-1][0][1] - 18))
            ui.text(f"true {tv:.3g}", (pos[-1][0][0] + rad + 10, pos[-1][0][1] + 2), DIM)
            verdict, vcol = f"Sample {self.sample + 1}/{len(d.X_te)}: predicted {pv:.3g}, true {tv:.3g} " \
                            f"(error {abs(pv - tv):.3g})", TEXT
        else:
            pred = int(self.predict_classes(out[None])[0])
            true = int(d.y_te[self.sample])
            if len(out) == 1:  # binary sigmoid: one neuron = P(class 1)
                probs = {0: 1 - out[0], 1: out[0]}
                p = pos[-1][0]
                ui.text(f"P({d.class_names[1]}) = {out[0] * 100:.1f}%", (p[0] + rad + 10, p[1] - 18))
                ui.text(f"pred {d.class_names[pred]}, true {d.class_names[true]}", (p[0] + rad + 10, p[1] + 2),
                        (120, 230, 120) if pred == true else (240, 120, 120))
            else:
                for p, idx in zip(pos[-1], shown[-1]):
                    if idx is None:
                        continue
                    tag = (" <- pred" if idx == pred else "") + (" [true]" if idx == true else "")
                    col = (120, 230, 120) if idx == pred == true else (240, 120, 120) if idx == pred else TEXT
                    ui.text(f"{d.class_names[idx][:14]} {out[idx] * 100:.1f}%{tag}", (p[0] + rad + 8, p[1]),
                            col, ui.small, anchor="midleft")
            ok = pred == true
            verdict = f"Sample {self.sample + 1}/{len(d.X_te)}: true = {d.class_names[true]}, " \
                      f"predicted = {d.class_names[pred]} ({'CORRECT' if ok else 'WRONG'})"
            vcol = (120, 230, 120) if ok else (240, 120, 120)

        status = "PAUSED" if self.paused else ("DONE" if self.epoch >= self.args.epochs else "TRAINING")
        if d.task == "regression":
            mtxt = f"Train MAE {self.metric:.3g}   Test MAE {self.val_metric:.3g}"
        else:
            mtxt = f"Train acc {self.metric * 100:.1f}%   Test acc {self.val_metric * 100:.1f}%"
        ui.text(f"{d.name} | Epoch {self.epoch}/{self.args.epochs}   Loss {self.loss:.4f}   {mtxt}   [{status}]",
                (20, H - 150), font=self.med)
        ui.text(verdict, (20, H - 115), vcol)
        ui.text("Lines: green = positive weight, red = negative, thicker = larger |w|.  Brightness = activation.",
                (20, H - 88), DIM, ui.small)
        ui.text("Controls: SPACE pause/resume   N next sample   R reset   B / Back button: wizard   ESC quit",
                (160, H - 45), DIM)
        ui.button((20, H - 60, 120, 40), "< Back", lambda m: self.go(3))
        self.loss_curve(pygame.Rect(W - 330, H - 170, 310, 110))

    def loss_curve(self, r):
        pygame.draw.rect(self.screen, (30, 33, 45), r)
        pygame.draw.rect(self.screen, DIM, r, 1)
        self.ui.text("Loss curve", (r.x + 6, r.y + 4), DIM, self.ui.small)
        if len(self.losses) < 2:
            return
        ls = np.array(self.losses[-300:])
        lo, hi = ls.min(), ls.max() + 1e-8
        pts = [(r.x + 5 + i * (r.w - 10) / (len(ls) - 1), r.bottom - 5 - (v - lo) / (hi - lo) * (r.h - 30))
               for i, v in enumerate(ls)]
        pygame.draw.lines(self.screen, (100, 180, 255), False, pts, 2)

    # ---------------- main loop ----------------
    def draw(self):
        self.screen.fill(BG)
        self.ui.begin()
        self.header()
        [self.draw_dataset, self.draw_architecture, self.draw_hidden, self.draw_output, self.draw_train][self.step]()
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
            if self.step == 4:
                if e.key == pygame.K_SPACE: self.paused = not self.paused
                elif e.key == pygame.K_n: self.sample = (self.sample + 1) % len(self.ds.X_te)
                elif e.key == pygame.K_r: self.build()
                elif e.key == pygame.K_b: self.go(3)
            elif e.key == pygame.K_RETURN:
                self.go(self.step + 1)
        return True

    def run(self):
        script = AutoScript(self) if self.args.auto else None
        running = True
        while running:
            for e in pygame.event.get():
                running = self.handle(e) and running
            if self.step == 4:
                self.train_step()
                if not self.paused and self.epoch and self.epoch % 10 == 0:
                    self.sample = (self.sample + 1) % len(self.ds.X_te)
            self.draw()
            if script and not script.tick():
                running = False
            self.clock.tick(30)
        pygame.quit()


class AutoScript:
    """Headless demo: walks the wizard with defaults, saves a screenshot per step, then trains."""

    def __init__(self, app):
        self.app, self.frame = app, 0
        self.out = app.args.screenshot_dir
        os.makedirs(self.out, exist_ok=True)
        self.train_frames = int(app.args.seconds * 30)

    def shot(self, name):
        path = os.path.join(self.out, name)
        pygame.image.save(self.app.screen, path)
        print("saved", path, flush=True)

    def tick(self):
        a, f = self.app, self.frame
        self.frame += 1
        if a.step < 4 and f % 5 == 4:  # let each step render a few frames
            if a.step == 2:  # demo group select: select all and apply tanh
                a.selected = set(range(len(a.hidden)))
                if a.hidden and a.hidden[0]["act"] != "tanh":
                    a.apply_selected("act", lambda _: "tanh")
                    return True
            self.shot(f"{a.step + 1}_{STEPS[a.step].lower().replace(' ', '_')}.png")
            a.go(a.step + 1)
            self.start = self.frame
        elif a.step == 4 and self.frame - self.start >= self.train_frames:
            self.shot(f"5_train_{a.ds_name.lower().replace(' ', '_')}.png")
            print(f"epoch {a.epoch} loss {a.loss:.4f} metric {a.val_metric:.4f}", flush=True)
            return False
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Iris", choices=list(DATASETS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--auto", action="store_true", help="scripted demo: step through wizard, train, save screenshots, exit")
    ap.add_argument("--seconds", type=float, default=8, help="training time in --auto mode")
    ap.add_argument("--screenshot-dir", default="screenshots")
    App(ap.parse_args()).run()
    sys.exit(0)
