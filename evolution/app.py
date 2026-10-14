"""Evolution mode: wizard (sensors/network, output, GA params) + live simulation view."""
import os
import time

import numpy as np
import pygame

from widgets import UI, BG, PANEL, TEXT, DIM, ACCENT, SEL
from render import draw_network, draw_line_chart
from .tracks import TRACKS, get_track, AREA
from .brain import Brain
from .ga import next_generation
from .sim import Sim, V_MAX, RADIUS

W, H = 1280, 780
ACTIVATIONS = ["relu", "tanh", "sigmoid", "elu", "leaky_relu", "selu", "linear"]
STEPS = ["Sensors & Network", "Output", "Evolution", "Run"]
OUT_MODES = [("tanh", "tanh", "Throttle tanh [-1,1] (reverse possible), steering tanh [-1,1]"),
             ("sigmoid", "tanh", "Throttle sigmoid [0,1] (forward only), steering tanh [-1,1]")]
SPEEDS = [1, 2, 4, 8, 16]
TRACK_OFS = (10, 60)


class EvoApp:
    def __init__(self, args, screen=None):
        self.args = args
        pygame.init()
        self.screen = screen or pygame.display.set_mode((W, H))
        pygame.display.set_caption("Neural Network Visualizer — Live Evolution")
        self.ui = UI(self.screen)
        self.clock = pygame.time.Clock()
        self.step = 0
        self.n_rays = 10
        self.speed_input = True
        self.hidden = [{"units": 8, "act": "tanh"}, {"units": 6, "act": "tanh"}]
        self.selected = {0, 1}
        self.out_mode = 0
        self.pop_size = getattr(args, "population", 50)
        self.mut_rate = 10   # percent of genes mutated
        self.mut_std = 30    # gaussian std * 100
        self.elitism = 4
        self.crossover = True
        self.gen_time = getattr(args, "gen_time", 20)
        self.track_name = getattr(args, "track", "Oval")
        self.result = "quit"
        self.status_msg = ""
        # Run-time state: initialized here so every code path (interactive or --auto)
        # can read it before a simulation has been started.
        self.running = False
        self.paused = False
        self.fast = False
        self.speed_i = 0
        self.sim = None
        self.brain = None
        self.pop = None
        self.acts = None
        self.gen = 0
        self.best_hist, self.avg_hist = [], []
        self.best_ever, self.best_genome = -1.0, None

    # ---------------- config helpers ----------------
    def sizes(self):
        return [self.n_rays + (1 if self.speed_input else 0)] + [h["units"] for h in self.hidden] + [2]

    def in_labels(self):
        labs = [f"ray {int(a)}°" for a in np.linspace(-90, 90, self.n_rays)]
        return labs + (["speed"] if self.speed_input else [])

    def set_n_hidden(self, n):
        n = max(0, min(6, n))
        while len(self.hidden) < n:
            self.hidden.append(dict(self.hidden[-1]) if self.hidden else {"units": 8, "act": "tanh"})
        self.hidden = self.hidden[:n]
        self.selected = {i for i in self.selected if i < n} or ({0} if n else set())

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

    # ---------------- evolution lifecycle ----------------
    def start_run(self):
        self.rng = np.random.default_rng(getattr(self.args, "seed", 42))
        self.track = get_track(self.track_name)
        thr_act, steer_act, _ = OUT_MODES[self.out_mode]
        self.brain = Brain(self.sizes(), [h["act"] for h in self.hidden], [thr_act, steer_act])
        self.sim = Sim(self.track, self.pop_size, self.n_rays, speed_input=self.speed_input,
                       throttle_mode=thr_act)
        self.pop = self.brain.random_genomes(self.pop_size, self.rng)
        self.gen = 1
        self.best_hist, self.avg_hist = [], []
        self.best_ever, self.best_genome = -1.0, self.pop[0].copy()
        self.paused, self.speed_i, self.fast = False, 0, False
        self.status_msg = ""

    def sim_tick(self):
        """Advance one physics step for the whole population; handle generation turnover."""
        x = self.sim.inputs()
        out, self.acts = self.brain.forward(self.pop, x, return_acts=True)
        self.sim.step(out)
        if not self.sim.alive.any() or self.sim.t >= self.gen_time:
            self.end_generation()

    def end_generation(self):
        fit = self.sim.fitness.copy()
        b = int(np.argmax(fit))
        self.best_hist.append(float(fit[b]))
        self.avg_hist.append(float(fit.mean()))
        if fit[b] > self.best_ever:
            self.best_ever, self.best_genome = float(fit[b]), self.pop[b].copy()
        print(f"GEN {self.gen} best={fit[b]:.2f} avg={fit.mean():.2f} "
              f"laps_best={int(self.sim.laps[b])} t={self.sim.t:.1f}s", flush=True)
        self.pop = next_generation(self.pop, fit, self.rng, elitism=self.elitism,
                                   mut_rate=self.mut_rate / 100, mut_std=self.mut_std / 100,
                                   crossover=self.crossover)
        self.gen += 1
        self.sim.reset()

    def save_best(self, path="best_genome.npz"):
        if self.best_genome is None:
            self.status_msg = "No completed generation yet - nothing to save."
            return
        np.savez(path, genome=self.best_genome, sizes=np.array(self.brain.sizes),
                 hidden_acts=np.array(self.brain.hidden_acts), out_acts=np.array(self.brain.out_acts),
                 fitness=self.best_ever, track=self.track_name)
        self.status_msg = f"Saved best genome (fitness {self.best_ever:.2f}) -> {path}"
        print(self.status_msg, flush=True)

    # ---------------- drawing: wizard ----------------
    def header(self):
        ui = self.ui
        ui.text("Live Evolution", (12, 14), font=ui.big)
        x = 230
        for i, s in enumerate(STEPS):
            col = ACCENT if i == self.step else (TEXT if i < self.step else DIM)
            r = ui.text(f"{i + 1}.{s}", (x, 18), col, ui.small)
            x = r.right + 18
        pygame.draw.line(self.screen, PANEL, (0, 50), (W, 50), 2)

    def nav(self, next_label="Next >"):
        ui = self.ui
        if self.step == 0:
            ui.button((20, H - 60, 120, 40), "< Menu", lambda m: self.to_menu())
        else:
            ui.button((20, H - 60, 120, 40), "< Back", lambda m: self.go(self.step - 1))
        ui.button((W - 160, H - 60, 140, 40), next_label, lambda m: self.go(self.step + 1), active=True)

    def to_menu(self):
        self.result = "menu"
        self.running = False

    def go(self, step):
        step = max(0, min(3, step))
        if step == 3:
            self.start_run()
        self.step = step

    def mini_network(self, rect):
        ui = self.ui
        sizes = self.sizes()
        names = ["sensors"] + [h["act"] for h in self.hidden] + ["/".join(OUT_MODES[self.out_mode][:2])]
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        n = len(sizes)
        for i, (s, nm) in enumerate(zip(sizes, names)):
            x = rect.x + 50 + i * (rect.w - 100) / max(n - 1, 1)
            hgt = min(rect.h - 60, 8 + 8 * min(s, 30))
            r = pygame.Rect(0, 0, 24, hgt); r.center = (int(x), rect.centery)
            pygame.draw.rect(self.screen, ACCENT if i in (0, n - 1) else SEL, r, border_radius=6)
            ui.text(s, (x, r.top - 16), anchor="center")
            ui.text(nm, (x, r.bottom + 12), DIM, ui.small, anchor="center")

    def draw_network_step(self):
        ui = self.ui
        ui.text("Step 1 — Sensors & network", (40, 66), font=ui.big)
        ui.spinner(40, 110, "Distance rays (180°)", self.n_rays,
                   lambda v: setattr(self, "n_rays", v), lo=3, hi=21)
        ui.button((360, 110, 230, 30), f"Speed input: {'ON' if self.speed_input else 'OFF'}",
                  lambda m: setattr(self, "speed_input", not self.speed_input), active=self.speed_input)
        ui.text(f"Inputs = {self.sizes()[0]}   Outputs = 2 (throttle, steering)", (620, 116), ACCENT)
        ui.spinner(40, 155, "Hidden layers", len(self.hidden), self.set_n_hidden, lo=0, hi=6)
        ui.text("Click to select; Ctrl-click toggles, Shift-click range. Edits apply to selection.",
                (40, 200), DIM, ui.small)
        for i, h in enumerate(self.hidden):
            ui.button((40, 225 + i * 40, 400, 34), f"Hidden {i + 1}: {h['units']} neurons, {h['act']}",
                      lambda m, i=i: self.toggle_select(i, m), active=i in self.selected)
        if self.hidden:
            yb = 225 + len(self.hidden) * 40 + 6
            ui.button((40, yb, 130, 30), "Select all",
                      lambda m: setattr(self, "selected", set(range(len(self.hidden)))))
            ui.button((180, yb, 130, 30), "Select none", lambda m: setattr(self, "selected", set()))
            x0 = 500
            first = self.hidden[min(self.selected)] if self.selected else self.hidden[0]
            ui.text("Neurons", (x0, 231))
            for k, d in enumerate([-10, -1, 1, 10]):
                ui.button((x0 + 100 + k * 58, 225, 52, 30), f"{d:+d}",
                          lambda m, d=d: self.apply_selected("units", lambda u: max(1, min(64, u + d))))
            ui.text("Activation", (x0, 276))
            for k, a in enumerate(ACTIVATIONS):
                ui.button((x0 + 100 + (k % 4) * 112, 270 + (k // 4) * 38, 106, 32), a,
                          lambda m, a=a: self.apply_selected("act", lambda _: a), active=a == first["act"])
        self.mini_network(pygame.Rect(40, 480, W - 80, 220))
        self.nav()

    def draw_output_step(self):
        ui = self.ui
        ui.text("Step 2 — Output layer (fixed: 2 neurons)", (40, 66), font=ui.big)
        ui.text("Output 0 = throttle (target speed), output 1 = steering angle.", (40, 105), DIM)
        for i, (t, s, desc) in enumerate(OUT_MODES):
            ui.button((40, 140 + i * 52, 760, 42), desc, lambda m, i=i: setattr(self, "out_mode", i),
                      active=i == self.out_mode)
        ui.text("No loss function: the network is trained by evolution (fitness), not gradients.",
                (40, 260), ACCENT)
        self.mini_network(pygame.Rect(40, 480, W - 80, 220))
        self.nav()

    def draw_evo_step(self):
        ui = self.ui
        ui.text("Step 3 — Evolution parameters & track", (40, 66), font=ui.big)
        ui.spinner(40, 110, "Population size", self.pop_size, lambda v: setattr(self, "pop_size", v), 4, 300, 2)
        ui.spinner(40, 150, "Mutation rate %", self.mut_rate, lambda v: setattr(self, "mut_rate", v), 1, 100)
        ui.spinner(40, 190, "Mutation std x100", self.mut_std, lambda v: setattr(self, "mut_std", v), 1, 300, 5)
        ui.spinner(40, 230, "Elitism (kept)", self.elitism,
                   lambda v: setattr(self, "elitism", min(v, self.pop_size)), 0, 50)
        ui.spinner(40, 270, "Max gen time (s)", self.gen_time, lambda v: setattr(self, "gen_time", v), 5, 120, 5)
        ui.button((40, 315, 300, 32), f"Crossover (uniform): {'ON' if self.crossover else 'OFF'}",
                  lambda m: setattr(self, "crossover", not self.crossover), active=self.crossover)
        ui.text("Selection: tournament (k=3). Mutation: gaussian noise on a random subset of genes.",
                (40, 360), DIM, ui.small)
        ui.text("Track", (40, 400))
        for i, name in enumerate(TRACKS):
            ui.button((120 + i * 130, 394, 120, 32), name, lambda m, n=name: setattr(self, "track_name", n),
                      active=name == self.track_name)
        # Track preview
        tr = get_track(self.track_name)
        prev = pygame.Rect(600, 70, 640, 560)
        pygame.draw.rect(self.screen, PANEL, prev, border_radius=8)
        sc = min(prev.w / AREA[0], prev.h / AREA[1])
        tf = lambda p: (prev.x + p[0] * sc, prev.y + p[1] * sc)
        self._draw_track(tr, tf, gates=True)
        ui.text(f"{tr.name}: {tr.n_gates} gates, width {tr.width}px", (prev.x + 10, prev.bottom - 22), DIM, ui.small)
        self.nav("Start >")

    def _draw_track(self, tr, tf, gates=False):
        poly_o = [tf(p) for p in tr.outer]
        poly_i = [tf(p) for p in tr.inner]
        pygame.draw.polygon(self.screen, (44, 48, 60), poly_o)
        pygame.draw.polygon(self.screen, BG, poly_i)
        if gates:
            for k, g in enumerate(tr.gates):
                if k % 2 == 0:
                    pygame.draw.line(self.screen, (60, 66, 84), tf(g[0]), tf(g[1]), 1)
        pygame.draw.lines(self.screen, (200, 200, 220), True, poly_o, 2)
        pygame.draw.lines(self.screen, (200, 200, 220), True, poly_i, 2)
        g0 = tr.gates[0]
        pygame.draw.line(self.screen, (240, 240, 240), tf(g0[0]), tf(g0[1]), 3)

    # ---------------- drawing: simulation ----------------
    def draw_run(self):
        ui, sim = self.ui, self.sim
        ox, oy = TRACK_OFS
        tf = lambda p: (ox + p[0], oy + p[1])
        alive = sim.alive
        best = int(np.argmax(np.where(alive, sim.fitness, -1e9))) if alive.any() else int(np.argmax(sim.fitness))

        if not self.fast:
            self._draw_track(self.track, tf, gates=True)
            # Best car rays
            for hp, d in zip(sim.ray_hits[best], sim.ray_dist[best]):
                c = (int(255 * (1 - d)), int(200 * d + 40), 80)
                pygame.draw.line(self.screen, c, tf(sim.pos[best]), tf(hp), 1)
                pygame.draw.circle(self.screen, c, (int(ox + hp[0]), int(oy + hp[1])), 3)
            order = np.argsort(alive.astype(int))  # dead first, alive on top
            for i in order:
                self._draw_car(i, tf, i == best)
        else:
            pygame.draw.rect(self.screen, PANEL, (ox, oy, AREA[0], AREA[1]), border_radius=8)
            ui.text("FAST-FORWARD (cars not rendered) — press F to show", (ox + AREA[0] / 2, oy + AREA[1] / 2),
                    ACCENT, ui.big, anchor="center")

        # Right panel: live network of best alive car
        x0 = 790
        nr = pygame.Rect(x0, 56, W - x0 - 10, 300)
        pygame.draw.rect(self.screen, (24, 27, 37), nr, border_radius=6)
        acts = [a[best] for a in self.acts] if self.acts is not None else \
            [np.zeros(s) for s in self.brain.sizes]
        weights = self.brain.weight_matrices(self.pop[best])
        names = ["in"] + [f"H{i + 1}" for i in range(len(self.hidden))] + ["out"]
        in_l = [f"{l} {v:.2f}" for l, v in zip(self.in_labels(), acts[0])]
        out_l = [f"throttle {acts[-1][0]:+.2f}", f"steer {acts[-1][1]:+.2f}"]
        draw_network(self.screen, ui, nr, self.brain.sizes, acts, weights, names, in_l, out_l, label_w=88)

        st = (f"Gen {self.gen}   t {sim.t:4.1f}/{self.gen_time}s   alive {int(alive.sum())}/{sim.n}   "
              f"speed x{SPEEDS[self.speed_i]}{'  FAST' if self.fast else ''}{'  PAUSED' if self.paused else ''}")
        ui.text(st, (x0, 364), TEXT)
        ui.text(f"Best now {sim.fitness[best]:.2f} (laps {int(sim.laps[best])})   "
                f"avg {sim.fitness.mean():.2f}   best ever {max(self.best_ever, 0):.2f}", (x0, 386), TEXT, ui.small)
        ui.text(f"Car #{best}: v={sim.v[best]:.0f}px/s  throttle={sim.out[best][0]:+.2f}  "
                f"steer={sim.out[best][1]:+.2f}", (x0, 404), DIM, ui.small)
        bi = int(np.argmax(self.best_hist)) + 1 if self.best_hist else None
        draw_line_chart(self.screen, ui, pygame.Rect(x0, 428, W - x0 - 10, 270),
                        [self.best_hist, self.avg_hist], ["Best", "Average"],
                        [(250, 210, 80), (100, 180, 255)], title="Fitness per generation",
                        xlabel="generation", ylabel="fitness", marker_x=bi, marker_label="best")
        if self.status_msg:
            ui.text(self.status_msg, (x0, 704), (120, 230, 120), ui.small)
        ui.button((20, H - 50, 110, 36), "< Menu", lambda m: self.to_menu())
        ui.button((140, H - 50, 110, 36), "< Config", lambda m: self.go(2))
        ui.text("SPACE pause   +/- sim speed   F fast-forward   S save best genome   ESC quit",
                (270, H - 40), DIM, ui.small)

    def _draw_car(self, i, tf, is_best):
        sim = self.sim
        p, h = sim.pos[i], sim.heading[i]
        c, s = np.cos(h), np.sin(h)
        L, Wd = 14, 7
        pts = [(L, Wd), (L, -Wd), (-L, -Wd), (-L, Wd)]
        poly = [tf((p[0] + x * c - y * s, p[1] + x * s + y * c)) for x, y in [(a / 1.4, b) for a, b in pts]]
        if not sim.alive[i]:
            col = (70, 60, 60)
        elif is_best:
            col = (250, 210, 80)
        else:
            hue = (i * 37) % 360
            col = pygame.Color(0); col.hsva = (hue, 55, 90, 100)
        pygame.draw.polygon(self.screen, col, poly)
        if is_best:
            pygame.draw.polygon(self.screen, (255, 255, 255), poly, 2)

    # ---------------- main loop ----------------
    def draw(self):
        self.screen.fill(BG)
        self.ui.begin()
        self.header()
        [self.draw_network_step, self.draw_output_step, self.draw_evo_step, self.draw_run][self.step]()
        pygame.display.flip()

    def handle(self, e):
        if e.type == pygame.QUIT:
            self.result = "quit"; return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
            mods = pygame.key.get_mods() | (pygame.KMOD_SHIFT if e.button == 3 else 0)
            self.ui.click(e.pos, mods)
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self.result = "quit"; return False
            if self.step == 3 and self.sim is not None:
                if e.key == pygame.K_SPACE: self.paused = not self.paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed_i = min(len(SPEEDS) - 1, self.speed_i + 1)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed_i = max(0, self.speed_i - 1)
                elif e.key == pygame.K_f: self.fast = not self.fast
                elif e.key == pygame.K_s: self.save_best()
                elif e.key == pygame.K_b: self.to_menu()
            elif e.key == pygame.K_RETURN:
                self.go(self.step + 1)
        return True

    def run(self):
        self.running = True
        script = EvoAuto(self) if self.args.auto else None
        while self.running:
            for e in pygame.event.get():
                if not self.handle(e):
                    self.running = False
            if self.step == 3 and self.sim is not None and not self.paused:
                if self.fast:
                    t0 = time.perf_counter()
                    while time.perf_counter() - t0 < 1 / 40:  # fill the frame budget with sim steps
                        self.sim_tick()
                else:
                    for _ in range(SPEEDS[self.speed_i]):
                        self.sim_tick()
            self.draw()
            if script and not script.tick():
                self.running = False
            self.clock.tick(0 if (script or self.fast) else 30)
        return self.result


class EvoAuto:
    """Headless demo: screenshot each wizard step, a live sim frame, then fast-forward generations."""

    def __init__(self, app):
        self.a, self.frame = app, 0
        self.out = app.args.screenshot_dir
        os.makedirs(self.out, exist_ok=True)
        self.target_gens = app.args.generations
        self.tag = app.track_name.lower()
        self.sim_shot = False
        self.resume_frame = 0

    def shot(self, name):
        path = os.path.join(self.out, name)
        pygame.image.save(self.a.screen, path)
        print("saved", path, flush=True)

    def tick(self):
        a, f = self.a, self.frame
        self.frame += 1
        if a.step < 3:
            if f % 3 == 2:
                self.shot(["e1_network.png", "e2_output.png", f"e3_evolution_{self.tag}.png"][a.step])
                a.go(a.step + 1)
            return True
        if not self.sim_shot and a.gen == 1 and a.sim.t > 3.0:
            a.speed_i = 1
            self.shot(f"e4_sim_running_{self.tag}.png")
            self.sim_shot = True
            a.fast = True
        if self.sim_shot and a.gen > self.target_gens:
            # Render a normal frame mid-generation to show evolved cars + fitness plot
            if a.fast:
                a.fast = False
                self.resume_frame = self.frame
            if self.frame - self.resume_frame >= 45:
                self.shot(f"e5_fitness_plot_{self.tag}.png")
                a.save_best(os.path.join(self.out, f"best_genome_{self.tag}.npz"))
                print("FITNESS_BEST " + " ".join(f"{v:.2f}" for v in a.best_hist), flush=True)
                print("FITNESS_AVG  " + " ".join(f"{v:.2f}" for v in a.avg_hist), flush=True)
                return False
        return True
