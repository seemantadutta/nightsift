"""Stage 2: fast keyboard-driven blink viewer (OpenCV window)."""
import ctypes
import os
import threading
import time
import cv2
import numpy as np

from .fitsio import read_fits, read_crop
from .metrics import make_preview

HELP = [
    'Right / D / Space  next frame        Left / A   previous',
    'X / Del            toggle reject      P          play / pause',
    '+ / -              faster / slower    Tab / ]    next filter group   [  previous group',
    'F                  view: all > flagged > suspect > kept > rejected  (yellow=suspect, red=rejected)',
    'S                  stretch: linked (per filter) / per frame',
    'Click              1:1 zoom at point  Z          toggle 1:1 zoom (centre)',
    'I                  info bar on/off    H          this help',
    'Q / Esc            quit (asks before moving files)',
]
K_LEFT, K_RIGHT, K_UP, K_DOWN, K_DEL = 2424832, 2555904, 2490368, 2621440, 3014656
WIN = 'NightSift'


def _mtf(m, x):
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def stf_lut(med, mad, target=0.25):
    """PixInsight-style auto-stretch as a 65536-entry uint16->uint8 lookup table."""
    med, mad = med / 65535.0, mad / 65535.0
    c0 = max(0.0, med - 2.8 * 1.4826 * mad)
    m = _mtf(target, max(med - c0, 1e-6))
    x = np.clip((np.arange(65536, dtype=np.float32) / 65535.0 - c0) / (1 - c0), 0, 1)
    return (np.clip(_mtf(m, x), 0, 1) * 255).astype(np.uint8)


def _screen_size():
    try:
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


class Blinker:
    def __init__(self, store, frames, view='all'):
        """frames: list of dicts with name, path, group, reasons, metrics (display order)."""
        self.store, self.all = store, frames
        self.view = 'all'
        self.frames = list(frames)
        self.i = 0
        self.prev = {}        # name -> uint16 preview
        self.stats = {}       # name -> (median, mad) of preview
        self.crops = {}       # (name, cx, cy) -> float crop
        self.linked, self.show_info, self.show_help = True, True, False
        self.playing, self.delay = False, 0.25
        self.zoom = None      # (cx, cy) fractions or None
        self._stop = False
        self.set_view(view)
        threading.Thread(target=self._preload, daemon=True).start()

    # -- state ------------------------------------------------------------
    def rejected(self, f):
        return self.store.is_rejected(f)

    def toggle(self, f):
        self.store.decisions[f['name']] = 'keep' if self.rejected(f) else 'reject'
        self.store.save()

    def set_view(self, v):
        cur = self.frames[self.i]['name'] if self.frames else None
        self.view = v
        if v == 'all':
            self.frames = list(self.all)
        elif v == 'suspect':
            self.frames = [f for f in self.all if f['tier'] == 'suspect']
        elif v == 'flagged':
            self.frames = [f for f in self.all if f['tier'] != 'ok' or self.rejected(f)]
        else:
            want = v == 'rejected'
            self.frames = [f for f in self.all if self.rejected(f) == want]
        names = [f['name'] for f in self.frames]
        self.i = names.index(cur) if cur in names else 0

    # -- image loading ----------------------------------------------------
    def _load(self, f):
        p = self.prev.get(f['name'])
        if p is None:
            p = self.store.load_preview(f['name'])
            if p is None:
                p = make_preview(read_fits(f['path'])[1])
                np.save(self.store.preview_path(f['name']), p)
            s = p[::4, ::4]
            med = float(np.median(s))
            self.stats[f['name']] = (med, float(np.median(np.abs(s - med))))
            self.prev[f['name']] = p
        return p

    def _preload(self):
        for f in self.all:
            if self._stop:
                return
            try:
                self._load(f)
            except Exception as e:
                print(f'  ! cannot load {f["name"]}: {e}')

    def _lut(self, f):
        if self.linked:
            ss = [self.stats[g['name']] for g in self.all if g['group'] == f['group'] and g['name'] in self.stats
                  and g['tier'] == 'ok']
            if ss:
                med, mad = np.median([s[0] for s in ss]), np.median([s[1] for s in ss])
                return stf_lut(med, mad)
        return stf_lut(*self.stats[f['name']])

    # -- rendering --------------------------------------------------------
    def render(self):
        if not self.frames:
            img = np.zeros((600, 1000, 3), np.uint8)
            cv2.putText(img, f'No frames in view "{self.view}" (press F)', (30, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return img
        f = self.frames[self.i]
        p = self._load(f)
        lut = self._lut(f)
        if self.zoom:
            key = (f['name'],) + self.zoom
            if key not in self.crops:
                if len(self.crops) > 64:
                    self.crops.clear()
                c = read_crop(f['path'], *self.zoom, cw=p.shape[1], ch=p.shape[0])
                self.crops[key] = np.clip(c, 0, 65535).astype(np.uint16)
            p = self.crops[key]
        img = cv2.cvtColor(lut[p], cv2.COLOR_GRAY2BGR)
        rej = self.rejected(f)
        if rej or f['tier'] == 'suspect':
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                          (0, 0, 255) if rej else (0, 210, 255), 12)
        if self.show_info:
            m = f['metrics']
            bar = np.zeros((64, img.shape[1], 3), np.uint8)
            t1 = (f'{self.i + 1}/{len(self.frames)} [{self.view}]  {f["group"]}   {f["name"]}')
            t2 = (f'stars {m.get("nstars", 0)}  HFR {m.get("hfr", float("nan")):.2f}  '
                  f'elong {m.get("elong", float("nan")):.2f}  bg {m.get("bg", 0):.0f}  '
                  f'{"LINKED" if self.linked else "per-frame"}  {"ZOOM 1:1" if self.zoom else ""}  '
                  f'{"PLAY %.2fs" % self.delay if self.playing else ""}')
            cv2.putText(bar, t1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(bar, t2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 180), 1, cv2.LINE_AA)
            d = self.store.decisions.get(f['name'])
            why = ', '.join(f['reasons'])
            if rej:
                txt, col = ('REJECT (you)' + (f': {why}' if why else '')) if d else f'REJECT: {why}', (60, 60, 255)
            elif f['tier'] == 'suspect':
                txt, col = (f'KEPT (you): {why}' if d else f'SUSPECT: {why}'), (0, 210, 255)
            elif f['tier'] == 'reject':
                txt, col = f'KEPT (you) despite: {why}', (0, 210, 255)
            else:
                txt, col = 'OK', (120, 200, 120)
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(bar, txt, (img.shape[1] - tw - 16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
            img = np.vstack([bar, img])
        if self.show_help:
            y = 110
            for line in HELP:
                cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
                y += 34
        return img

    # -- navigation -------------------------------------------------------
    def step(self, d):
        if self.frames:
            self.i = (self.i + d) % len(self.frames)

    def jump_group(self, d):
        if not self.frames:
            return
        g = self.frames[self.i]['group']
        groups = list(dict.fromkeys(f['group'] for f in self.frames))
        ng = groups[(groups.index(g) + d) % len(groups)]
        self.i = next(k for k, f in enumerate(self.frames) if f['group'] == ng)

    def _on_mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and self.frames:
            p = self.prev.get(self.frames[self.i]['name'])
            if p is None:
                return
            y -= 64 if self.show_info else 0
            if self.zoom:
                self.zoom = None
            else:
                self.zoom = (min(max(x / p.shape[1], 0), 1), min(max(y / p.shape[0], 0), 1))
            self.dirty = True

    def run(self):
        sw, sh = _screen_size()
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        first = self.render()
        s = min(0.92 * sw / first.shape[1], 0.88 * sh / first.shape[0], 1.0)
        cv2.resizeWindow(WIN, int(first.shape[1] * s), int(first.shape[0] * s))
        cv2.setMouseCallback(WIN, self._on_mouse)
        self.dirty, img, last = False, first, time.perf_counter()
        while True:
            cv2.imshow(WIN, img)
            wait = max(1, int((self.delay - (time.perf_counter() - last)) * 1000)) if self.playing else 50
            k = cv2.waitKeyEx(wait)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == -1:
                if self.playing and time.perf_counter() - last >= self.delay:
                    self.step(1)
                    last = time.perf_counter()
                    img = self.render()
                elif self.dirty or self.show_help:
                    self.dirty = False
                    img = self.render()
                continue
            c = chr(k).lower() if 0 <= k < 256 else ''
            if c in ('q', '\x1b'):
                break
            elif k == K_RIGHT or c in ('d', ' '):
                self.step(1)
            elif k == K_LEFT or c == 'a':
                self.step(-1)
            elif k == K_DEL or c == 'x':
                if self.frames:
                    self.toggle(self.frames[self.i])
            elif c == 'p':
                self.playing = not self.playing
                last = time.perf_counter()
            elif c in ('+', '='):
                self.delay = max(0.05, self.delay / 1.4)
            elif c in ('-', '_'):
                self.delay = min(3.0, self.delay * 1.4)
            elif c in ('\t', ']'):
                self.jump_group(1)
            elif c == '[':
                self.jump_group(-1)
            elif c == 'f':
                self.set_view({'all': 'flagged', 'flagged': 'suspect', 'suspect': 'kept', 'kept': 'rejected',
                               'rejected': 'all'}[self.view])
            elif c == 's':
                self.linked = not self.linked
            elif c == 'z':
                self.zoom = None if self.zoom else (0.5, 0.5)
            elif c == 'i':
                self.show_info = not self.show_info
            elif c == 'h':
                self.show_help = not self.show_help
            img = self.render()
        self._stop = True
        cv2.destroyAllWindows()
