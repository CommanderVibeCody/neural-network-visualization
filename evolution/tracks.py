"""Hand-made closed tracks. Each track is a centerline polyline plus width; inner/outer
walls are derived by offsetting along normals. Gates (checkpoints) span inner->outer."""
import numpy as np

AREA = (760, 660)  # track coordinate space (pixels)


def catmull_rom_closed(ctrl, samples_per_seg=12):
    """Smooth closed curve through control points."""
    p = np.asarray(ctrl, dtype=float)
    n = len(p)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def _seg_x(a, b, c, d):
    """Intersection point of segments ab and cd, or None."""
    r, q = b - a, d - c
    den = r[0] * q[1] - r[1] * q[0]
    if abs(den) < 1e-12:
        return None
    t = ((c - a)[0] * q[1] - (c - a)[1] * q[0]) / den
    u = ((c - a)[0] * r[1] - (c - a)[1] * r[0]) / den
    if 0 < t < 1 and 0 < u < 1:
        return a + t * r
    return None


def remove_loops(poly, window=12):
    """Remove small self-intersection loops that appear when offsetting a curve at tight bends."""
    pts = [np.asarray(p, float) for p in poly]
    changed = True
    while changed:
        changed = False
        n = len(pts)
        for i in range(n):
            for k in range(2, window):
                j = (i + k) % n
                x = _seg_x(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n])
                if x is None:
                    continue
                drop = {(i + m) % n for m in range(1, k + 1)}
                new_pts = []
                for m, p in enumerate(pts):
                    if m in drop:
                        continue
                    new_pts.append(p)
                    if m == i:
                        new_pts.append(x)
                pts = new_pts
                changed = True
                break
            if changed:
                break
    return np.array(pts)


class Track:
    def __init__(self, name, center, width):
        self.name = name
        c = np.asarray(center, dtype=float)
        self.center = c
        self.width = width
        nxt = np.roll(c, -1, axis=0)
        prv = np.roll(c, 1, axis=0)
        tang = nxt - prv
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        inner_raw = c - normal * width / 2
        outer_raw = c + normal * width / 2
        # Gates use the raw (per-centerline-point) offsets so they stay 1:1 with the centerline.
        self.gates = np.stack([inner_raw, outer_raw], axis=1)  # (G, 2, 2)
        # Walls are cleaned of tiny loops at tight bends.
        self.inner = remove_loops(inner_raw)
        self.outer = remove_loops(outer_raw)
        # Wall segments as (M, 4): x1, y1, x2, y2
        segs = []
        for wall in (self.inner, self.outer):
            a, b = wall, np.roll(wall, -1, axis=0)
            segs.append(np.hstack([a, b]))
        self.walls = np.vstack(segs)
        self.gate_mid = c
        self.n_gates = len(c)
        d = c[1] - c[0]
        self.start_pos = c[0] + 0.35 * d
        self.start_heading = float(np.arctan2(d[1], d[0]))


def _oval():
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    cx, cy, a, b = 380, 330, 300, 220
    # Clockwise in screen space (y down), slight squircle for straights
    x = cx + a * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** 0.7
    y = cy + b * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** 0.7
    return Track("Oval", np.stack([x, y], axis=1), 95)


def _curvy():
    ctrl = [(110, 330), (130, 140), (260, 80), (380, 160), (510, 85), (660, 130), (690, 290),
            (610, 370), (680, 490), (600, 600), (440, 590), (350, 500), (240, 600), (100, 520)]
    return Track("Curvy", catmull_rom_closed(ctrl, 7), 70)


def _kidney():
    ctrl = [(110, 200), (300, 80), (520, 90), (680, 220), (670, 440), (540, 590),
            (390, 520), (240, 600), (90, 470)]
    return Track("Kidney", catmull_rom_closed(ctrl, 9), 80)


TRACKS = {"Oval": _oval, "Curvy": _curvy, "Kidney": _kidney}


def get_track(name):
    return TRACKS[name]()
