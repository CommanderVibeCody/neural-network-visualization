"""Minimal self-drawn Pygame widgets (immediate-mode style)."""
import pygame

BG = (18, 20, 28)
PANEL = (32, 36, 50)
TEXT = (220, 220, 230)
DIM = (140, 140, 160)
ACCENT = (90, 150, 255)
SEL = (60, 90, 160)


class UI:
    """Widgets are re-declared every frame; clicks are matched against last frame's rects."""

    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("dejavusans", 15)
        self.small = pygame.font.SysFont("dejavusans", 12)
        self.big = pygame.font.SysFont("dejavusans", 22, bold=True)
        self.widgets = []        # (rect, callback) declared this frame
        self.prev = []
        self.mouse = (-1, -1)

    def begin(self):
        self.prev, self.widgets = self.widgets, []
        self.mouse = pygame.mouse.get_pos()

    def click(self, pos, mods=0):
        # Iterate in reverse so widgets drawn later (on top) win.
        for rect, cb in reversed(self.prev):
            if rect.collidepoint(pos):
                cb(mods)
                return True
        return False

    def text(self, s, xy, color=TEXT, font=None, anchor="topleft"):
        surf = (font or self.font).render(str(s), True, color)
        r = surf.get_rect(**{anchor: xy})
        self.screen.blit(surf, r)
        return r

    def button(self, rect, label, cb, active=False, enabled=True):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(self.mouse) and enabled
        col = ACCENT if active else (SEL if hover else PANEL)
        if not enabled:
            col = (26, 28, 36)
        pygame.draw.rect(self.screen, col, rect, border_radius=6)
        pygame.draw.rect(self.screen, DIM, rect, 1, border_radius=6)
        self.text(label, rect.center, TEXT if enabled else DIM, anchor="center")
        if enabled:
            self.widgets.append((rect, cb))  # cb(mods)
        return rect

    def spinner(self, x, y, label, value, on_change, lo=1, hi=512, step=1):
        """[-] value [+]; shift-click changes by 10x step."""
        self.text(label, (x, y + 6))
        bx = x + 170
        self.button((bx, y, 30, 30), "-", lambda m: on_change(max(lo, value - step * (10 if m & pygame.KMOD_SHIFT else 1))))
        self.text(value, (bx + 65, y + 15), anchor="center")
        self.button((bx + 100, y, 30, 30), "+", lambda m: on_change(min(hi, value + step * (10 if m & pygame.KMOD_SHIFT else 1))))

    def cycle(self, rect, options, current, on_change, fmt=str):
        """Dropdown-ish button cycling through options (right-click / shift goes backwards)."""
        i = options.index(current) if current in options else 0
        def nxt(m):
            d = -1 if m & pygame.KMOD_SHIFT else 1
            on_change(options[(i + d) % len(options)])
        return self.button(rect, f"< {fmt(current)} >", nxt)

