"""Shared drawing helpers: network diagram and line chart with grid / numeric axes."""
import numpy as np
import pygame

from widgets import BG, DIM, TEXT, ACCENT

POS_W, NEG_W = (60, 200, 90), (220, 70, 70)
MAX_DRAWN = 16


def shown_indices(n, cap=MAX_DRAWN):
    if n <= cap:
        return list(range(n))
    half = (cap - 1) // 2
    return list(range(half)) + [None] + list(range(n - (cap - 1 - half), n))


def draw_network(screen, ui, rect, sizes, acts, weights, layer_names=None,
                 in_labels=None, out_labels=None, label_w=110):
    """Draw layers as circles (brightness = activation) and weights as colored lines.
    acts: list of 1-D arrays per layer; weights: list of (in, out) kernels."""
    left, right = rect.x + label_w, rect.right - label_w
    top, bottom = rect.y + 22, rect.bottom - 8
    nl = len(sizes)
    shown = [shown_indices(n) for n in sizes]
    pos = []
    for i, idx in enumerate(shown):
        x = left + i * (right - left) / max(nl - 1, 1)
        gap = (bottom - top) / len(idx)
        pos.append([(int(x), int(top + gap * (j + 0.5))) for j in range(len(idx))])
    rad = int(max(4, min(14, (bottom - top) / max(len(s) for s in shown) / 2 - 2)))

    for li, w in enumerate(weights):
        m = np.max(np.abs(w)) + 1e-8
        for a, ia in enumerate(shown[li]):
            for b, ib in enumerate(shown[li + 1]):
                if ia is None or ib is None:
                    continue
                v = w[ia, ib] / m
                k = abs(v)
                base = POS_W if v > 0 else NEG_W
                col = tuple(int(BG[c] + (base[c] - BG[c]) * (0.1 + 0.9 * k)) for c in range(3))
                pygame.draw.line(screen, col, pos[li][a], pos[li + 1][b], max(1, int(1 + 2.5 * k)))

    for li, a in enumerate(acts):
        a = np.asarray(a, dtype=float)
        if li == 0:
            norm = np.clip(a, 0, 1)
        else:
            norm = np.clip(np.abs(a) / (np.max(np.abs(a)) + 1e-8), 0, 1)
        for p, idx in zip(pos[li], shown[li]):
            if idx is None:
                ui.text("...", p, DIM, ui.small, anchor="center")
                continue
            g = float(norm[idx])
            col = (int(30 + 225 * g), int(40 + 180 * g), int(70 + 60 * g))
            pygame.draw.circle(screen, col, p, rad)
            pygame.draw.circle(screen, (200, 200, 220), p, rad, 1)
        if layer_names:
            ui.text(f"{layer_names[li]}[{sizes[li]}]", (pos[li][0][0], rect.y + 8), DIM, ui.small,
                    anchor="center")
    if in_labels:
        for p, idx in zip(pos[0], shown[0]):
            if idx is not None and idx < len(in_labels):
                ui.text(in_labels[idx], (p[0] - rad - 4, p[1]), TEXT, ui.small, anchor="midright")
    if out_labels:
        for p, idx in zip(pos[-1], shown[-1]):
            if idx is not None and idx < len(out_labels):
                ui.text(out_labels[idx], (p[0] + rad + 4, p[1]), TEXT, ui.small, anchor="midleft")
    return pos


def draw_line_chart(screen, ui, r, series, labels, colors, title="", xlabel="x", ylabel="y",
                    log_scale=False, nonneg=True, marker_x=None, marker_label=""):
    """Line chart with grid, numeric y ticks, integer x ticks (1-based), end-value labels, legend."""
    pygame.draw.rect(screen, (30, 33, 45), r)
    pygame.draw.rect(screen, DIM, r, 1)
    ui.text(title, (r.x + 8, r.y + 4), DIM, ui.small)
    pad_l, pad_r, pad_t, pad_b = 52, 56, 34, 32
    plot = pygame.Rect(r.x + pad_l, r.y + pad_t, r.w - pad_l - pad_r, r.h - pad_t - pad_b)
    usable = [np.asarray(s, float) for s in series if len(s)]
    if not usable:
        ui.text("Waiting for data…", plot.center, DIM, anchor="center")
        return
    n = max(len(s) for s in series)
    allv = np.concatenate(usable)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return
    fwd = (lambda v: np.log10(max(float(v), 1e-6))) if log_scale else float
    inv = (lambda u: 10 ** u) if log_scale else (lambda u: u)
    lo, hi = fwd(allv.min()), fwd(allv.max())
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    if nonneg and not log_scale and allv.min() >= 0:
        lo = max(0.0, lo)

    def y_of(v):
        return plot.bottom - (fwd(v) - lo) / (hi - lo) * plot.h

    def x_of(i):
        return plot.x + (i / max(n - 1, 1)) * plot.w

    for ti in range(6):
        u = lo + (hi - lo) * ti / 5
        yy = plot.bottom - ti / 5 * plot.h
        pygame.draw.line(screen, (48, 52, 68), (plot.x, yy), (plot.right, yy), 1)
        v = inv(u)
        lab = f"{v:.0e}" if (log_scale or (0 < abs(v) < 0.01)) else (f"{v:.3f}" if abs(v) < 1 else f"{v:.3g}")
        ui.text(lab, (plot.x - 5, yy), DIM, ui.small, anchor="midright")
    n_xt = min(6, max(2, n))
    for ti in range(n_xt):
        xi = int(round(ti * (n - 1) / max(n_xt - 1, 1)))
        xx = x_of(xi)
        pygame.draw.line(screen, (48, 52, 68), (xx, plot.y), (xx, plot.bottom), 1)
        ui.text(str(xi + 1), (xx, plot.bottom + 3), DIM, ui.small, anchor="midtop")
    pygame.draw.rect(screen, DIM, plot, 1)
    ui.text(xlabel, (plot.centerx, r.bottom - 10), DIM, ui.small, anchor="center")
    ui.text(ylabel, (r.x + 4, r.y + 20), DIM, ui.small)

    for s, col in zip(series, colors):
        if not len(s):
            continue
        pts = [(x_of(i), y_of(v)) for i, v in enumerate(s)]
        if len(pts) >= 2:
            pygame.draw.lines(screen, col, False, pts, 2)
        lx, ly = pts[-1]
        pygame.draw.circle(screen, col, (int(lx), int(ly)), 3)
        ui.text(f"{s[-1]:.3g}", (lx + 5, ly), col, ui.small, anchor="midleft")
    if marker_x and n > 1:
        bx = x_of(marker_x - 1)
        pygame.draw.line(screen, ACCENT, (bx, plot.y), (bx, plot.bottom), 1)
        ui.text(marker_label, (bx + 3, plot.y + 2), ACCENT, ui.small)
    lx = r.right - 10
    for lab, col in reversed(list(zip(labels, colors))):
        rr = ui.text(lab, (lx, r.y + 4), DIM, ui.small, anchor="topright")
        pygame.draw.line(screen, col, (rr.x - 18, r.y + 11), (rr.x - 4, r.y + 11), 2)
        lx = rr.x - 26
