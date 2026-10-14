"""Live neural network visualizer: trains a small tf.keras model on Iris
and draws all layers, activations and weights in a Pygame window."""
import os
import sys
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import pygame
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

W, H = 1200, 750
BG = (18, 20, 28)
TEXT = (220, 220, 230)
DIM = (140, 140, 160)
LAYER_SIZES = [4, 8, 6, 3]
LAYER_NAMES = ["Input", "Hidden 1 (ReLU)", "Hidden 2 (ReLU)", "Output (Softmax)"]


def load_data():
    """Load Iris into a pandas DataFrame, split and standardize it."""
    iris = load_iris(as_frame=True)
    df: pd.DataFrame = iris.frame
    features = list(iris.feature_names)
    X = df[features].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    return (scaler.transform(X_tr).astype(np.float32), y_tr,
            scaler.transform(X_te).astype(np.float32), y_te, X_te,
            features, list(iris.target_names))


def build_model():
    """4 -> 8 relu -> 6 relu -> 3 softmax, plus a probe model exposing every layer output."""
    inp = tf.keras.Input(shape=(4,))
    h1 = tf.keras.layers.Dense(8, activation="relu")(inp)
    h2 = tf.keras.layers.Dense(6, activation="relu")(h1)
    out = tf.keras.layers.Dense(3, activation="softmax")(h2)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    probe = tf.keras.Model(inp, [h1, h2, out])
    return model, probe


class App:
    def __init__(self, max_epochs=300, batch_size=16):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        (self.X_tr, self.y_tr, self.X_te, self.y_te, self.X_te_raw,
         self.features, self.classes) = load_data()
        pygame.init()
        pygame.display.set_caption("Neural Network Visualizer (Keras + Pygame)")
        self.screen = pygame.display.set_mode((W, H))
        self.font = pygame.font.SysFont("dejavusans", 15)
        self.big = pygame.font.SysFont("dejavusans", 20, bold=True)
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        tf.keras.backend.clear_session()
        self.model, self.probe = build_model()
        self.epoch, self.loss, self.acc, self.val_acc = 0, 0.0, 0.0, 0.0
        self.losses = []
        self.sample = 0
        self.paused = False
        self.rng = np.random.default_rng()

    def train_step(self):
        """One epoch of mini-batch training via train_on_batch (keeps GUI responsive)."""
        if self.paused or self.epoch >= self.max_epochs:
            return
        idx = self.rng.permutation(len(self.X_tr))
        losses, accs = [], []
        for s in range(0, len(idx), self.batch_size):
            b = idx[s:s + self.batch_size]
            l, a = self.model.train_on_batch(self.X_tr[b], self.y_tr[b])
            losses.append(float(l)); accs.append(float(a))
        self.epoch += 1
        self.loss, self.acc = float(np.mean(losses)), float(np.mean(accs))
        self.losses.append(self.loss)
        pred = np.argmax(self.model(self.X_te, training=False).numpy(), axis=1)
        self.val_acc = float(np.mean(pred == self.y_te))

    # ---------- drawing ----------
    def neuron_positions(self):
        left, right, top, bottom = 220, 800, 90, 560
        pos = []
        for i, n in enumerate(LAYER_SIZES):
            x = left + i * (right - left) / (len(LAYER_SIZES) - 1)
            gap = (bottom - top) / max(n, 1)
            pos.append([(int(x), int(top + gap * (j + 0.5))) for j in range(n)])
        return pos

    def text(self, s, xy, color=TEXT, font=None, anchor="topleft"):
        surf = (font or self.font).render(s, True, color)
        r = surf.get_rect(**{anchor: xy})
        self.screen.blit(surf, r)

    def draw(self):
        self.screen.fill(BG)
        x = self.X_te[self.sample:self.sample + 1]
        acts = [x[0]] + [a.numpy()[0] for a in self.probe(x, training=False)]
        weights = [l.get_weights()[0] for l in self.model.layers if l.get_weights()]
        pos = self.neuron_positions()

        # Connections: green = positive, red = negative, thickness/alpha = magnitude
        for li, w in enumerate(weights):
            m = np.max(np.abs(w)) + 1e-8
            for i, p0 in enumerate(pos[li]):
                for j, p1 in enumerate(pos[li + 1]):
                    v = w[i, j] / m
                    k = abs(v)
                    base = (60, 200, 90) if v > 0 else (220, 70, 70)
                    col = tuple(int(BG[c] + (base[c] - BG[c]) * (0.15 + 0.85 * k)) for c in range(3))
                    pygame.draw.line(self.screen, col, p0, p1, max(1, int(1 + 4 * k)))

        # Neurons: brightness = activation (normalized per layer)
        for li, a in enumerate(acts):
            a = np.asarray(a, dtype=np.float32)
            if li == 0:
                norm = np.clip((a + 2.5) / 5.0, 0, 1)  # standardized inputs
            elif li == len(acts) - 1:
                norm = a  # softmax probabilities
            else:
                norm = a / (np.max(a) + 1e-8)
            for j, p in enumerate(pos[li]):
                g = float(norm[j])
                col = (int(30 + 225 * g), int(40 + 180 * g), int(70 + 60 * g))
                pygame.draw.circle(self.screen, col, p, 18)
                pygame.draw.circle(self.screen, (200, 200, 220), p, 18, 2)
                self.text(f"{a[j]:.2f}", p, (0, 0, 0) if g > 0.6 else TEXT, anchor="center")
            self.text(LAYER_NAMES[li], (pos[li][0][0], 60), DIM, anchor="center")

        # Input labels: feature names + raw values
        raw = self.X_te_raw[self.sample]
        for j, p in enumerate(pos[0]):
            self.text(f"{self.features[j]}", (p[0] - 28, p[1] - 10), TEXT, anchor="topright")
            self.text(f"= {raw[j]:.1f}", (p[0] - 28, p[1] + 6), DIM, anchor="topright")

        # Output labels: class names, predicted vs true
        probs = acts[-1]
        pred, true = int(np.argmax(probs)), int(self.y_te[self.sample])
        for j, p in enumerate(pos[-1]):
            tag = ""
            if j == pred: tag += "  <- predicted"
            if j == true: tag += "  [true]"
            col = (120, 230, 120) if (j == pred and pred == true) else (240, 120, 120) if j == pred else TEXT
            self.text(f"{self.classes[j]} {probs[j]*100:.1f}%{tag}", (p[0] + 28, p[1] - 8), col)

        # Stats panel
        status = "PAUSED" if self.paused else ("DONE" if self.epoch >= self.max_epochs else "TRAINING")
        self.text(f"Epoch {self.epoch}/{self.max_epochs}   Loss {self.loss:.4f}   "
                  f"Train acc {self.acc*100:.1f}%   Test acc {self.val_acc*100:.1f}%   [{status}]",
                  (20, 15), font=self.big)
        verdict = "CORRECT" if pred == true else "WRONG"
        self.text(f"Test sample {self.sample + 1}/{len(self.X_te)}: true = {self.classes[true]}, "
                  f"predicted = {self.classes[pred]} ({verdict})", (20, 590),
                  (120, 230, 120) if pred == true else (240, 120, 120))
        self.text("Controls: SPACE pause/resume   N next sample   R reset   ESC quit",
                  (20, H - 30), DIM)
        self.text("Lines: green = positive weight, red = negative, thicker = larger |w|. "
                  "Neuron brightness = activation.", (20, H - 55), DIM)
        self.draw_loss_curve(pygame.Rect(W - 330, 600, 310, 110))
        pygame.display.flip()

    def draw_loss_curve(self, r):
        pygame.draw.rect(self.screen, (30, 33, 45), r)
        pygame.draw.rect(self.screen, DIM, r, 1)
        self.text("Loss curve", (r.x + 6, r.y + 4), DIM)
        if len(self.losses) < 2:
            return
        ls = np.array(self.losses)
        lo, hi = ls.min(), ls.max() + 1e-8
        pts = [(r.x + 5 + i * (r.w - 10) / (len(ls) - 1),
                r.bottom - 5 - (v - lo) / (hi - lo) * (r.h - 30)) for i, v in enumerate(ls)]
        pygame.draw.lines(self.screen, (100, 180, 255), False, pts, 2)

    def run(self, max_seconds=None, screenshot=None):
        start = pygame.time.get_ticks()
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE: running = False
                    elif e.key == pygame.K_SPACE: self.paused = not self.paused
                    elif e.key == pygame.K_n: self.sample = (self.sample + 1) % len(self.X_te)
                    elif e.key == pygame.K_r: self.reset()
            self.train_step()
            # Auto-advance the displayed sample every 10 epochs while training
            if not self.paused and self.epoch and self.epoch % 10 == 0 and self.epoch < self.max_epochs:
                self.sample = (self.sample + 1) % len(self.X_te)
            self.draw()
            self.clock.tick(30)
            if max_seconds and (pygame.time.get_ticks() - start) / 1000 > max_seconds:
                running = False
        if screenshot:
            pygame.image.save(self.screen, screenshot)
            print(f"Screenshot saved to {screenshot} (epoch {self.epoch}, loss {self.loss:.4f})")
        pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=None, help="auto-quit after N seconds")
    ap.add_argument("--screenshot", default=None, help="save PNG of the window on exit")
    a = ap.parse_args()
    App(max_epochs=a.epochs).run(a.seconds, a.screenshot)
    sys.exit(0)
