"""Timeline: one quality metric per frame across the night(s).

Dots are coloured by filter (LRGB as-is, narrowband in the Hubble palette: SII red, Ha green,
OIII blue) and ringed by status (red = reject, yellow = suspect).
Dashed lines show the reject / suspect limits at the current strictness; the bad side of each
line is shaded, so good frames are always the dots in the unshaded area.
Click a dot to jump to that frame; drag across a range to select several frames.
"""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from .scoring import night_of

_SII, _HA, _OIII = QColor(230, 70, 70), QColor(80, 200, 100), QColor(80, 140, 255)   # Hubble (SHO) palette
FILTER_COLORS = {'L': QColor(225, 225, 225), 'R': QColor(235, 80, 70), 'G': QColor(90, 200, 90),
                 'B': QColor(80, 140, 255),
                 'SII': _SII, 'S': _SII, 'S2': _SII, 'HA': _HA, 'H': _HA, 'H-ALPHA': _HA,
                 'OIII': _OIII, 'O': _OIII, 'O3': _OIII}
RING = {'reject': QColor(255, 40, 40), 'suspect': QColor(255, 200, 0)}

# name -> (label, value getter, limits getter(limits_pair) -> (reject, suspect) or None, lower_is_bad, fmt)
METRICS = {
    'weight': ('Weight', lambda f: f['weight'] * 100, lambda L: (L[0]['weight'] * 100, L[1]['weight'] * 100),
               True, '{:.0f}%'),
    'fwhm_r': ('FWHM % of group', lambda f: f['r_fwhm'] * 100, lambda L: (L[0]['fwhm'] * 100, L[1]['fwhm'] * 100),
               False, '{:.0f}%'),
    'fwhm': ('FWHM ″', lambda f: f['metrics'].get('fwhm', np.nan) * f['metrics'].get('scale', np.nan), None,
             False, '{:.1f}″'),
    'hfr_r': ('HFR % of group', lambda f: f['r_hfr'] * 100, lambda L: (L[0]['hfr'] * 100, L[1]['hfr'] * 100),
              False, '{:.0f}%'),
    'sky': ('Sky × group', lambda f: f['r_sky'], None, False, '{:.2f}×'),
    'stars': ('Stars % of group', lambda f: f['r_stars'] * 100, None, True, '{:.0f}%'),
    'ecc': ('Eccentricity', lambda f: f['metrics'].get('ecc', np.nan), lambda L: (L[0]['ecc'], L[1]['ecc']),
            False, '{:.2f}'),
}


# upper display limit per metric: the few frames above it are pinned to the top edge (▲)
CAPS = {'weight': 150.0}


def filter_color(name):
    return FILTER_COLORS.get((name or '').upper(), QColor(200, 200, 120))


class Timeline(QWidget):
    frame_clicked = Signal(str)
    selection_changed = Signal(list)

    ML, MR, MT, MB = 56, 12, 10, 30   # margins

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.frames, self.status, self.metric_key, self.limits, self.current = [], {}, 'weight', None, None
        self.selected = set()
        self._pts = []            # (x, y, frame) in widget coords
        self._drag = None         # (x0, x1) during drag

    def set_data(self, frames, status, limits, current=None):
        """frames: dicts with metrics + r_* + weight; status: name -> 'reject'|'suspect'|'ok'."""
        self.frames = sorted(frames, key=lambda f: f['metrics'].get('date', ''))
        self.status, self.limits, self.current = status, limits, current
        names = {f['name'] for f in self.frames}
        self.selected &= names
        self.update()

    def set_metric(self, key):
        self.metric_key = key
        self.update()

    def set_current(self, name):
        self.current = name
        self.update()

    def clear_selection(self):
        self.selected.clear()
        self.selection_changed.emit([])
        self.update()

    # -- geometry ----------------------------------------------------------------------------
    def _plot_rect(self):
        return QRectF(self.ML, self.MT, max(10, self.width() - self.ML - self.MR),
                      max(10, self.height() - self.MT - self.MB))

    def _y_range(self, vals, lims):
        v = vals[np.isfinite(vals)]
        if not len(v):
            return 0.0, 1.0
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        if lims:
            lo, hi = min(lo, *lims), max(hi, *lims)
        cap = CAPS.get(self.metric_key)
        if cap is not None:
            hi = min(hi, max(cap, (max(lims) * 1.3) if lims else cap))
        if METRICS[self.metric_key][3]:          # lower is bad: always show from 0
            lo = 0.0
        pad = (hi - lo) * 0.08 or 1.0
        return lo - pad * (0 if METRICS[self.metric_key][3] else 1), hi + pad

    # -- painting ----------------------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 24, 27))
        r = self._plot_rect()
        label, get, lim_fn, low_bad, fmt = METRICS[self.metric_key]
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1))
        p.setFont(small)
        if not self.frames:
            p.setPen(QColor(130, 130, 130))
            p.drawText(r, Qt.AlignCenter, 'No frames yet')
            return
        vals = np.array([get(f) for f in self.frames], dtype=float)
        lims = lim_fn(self.limits) if (lim_fn and self.limits) else None
        y0, y1 = self._y_range(vals, lims)
        n = len(self.frames)
        xs = r.left() + (np.arange(n) + 0.5) * r.width() / n

        def ymap(v):
            return r.bottom() - (np.clip(v, y0, y1) - y0) / (y1 - y0) * r.height()

        # night separators + labels, hour ticks
        p.setPen(QPen(QColor(70, 70, 76), 1))
        nights = [night_of(f['metrics']) for f in self.frames]
        last_hour, last_x = None, -1e9
        for i, f in enumerate(self.frames):
            if i == 0 or nights[i] != nights[i - 1]:
                x = xs[i] - r.width() / n / 2
                p.setPen(QPen(QColor(110, 110, 120), 1))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                p.setPen(QColor(160, 160, 170))
                p.drawText(QPointF(x + 4, r.top() + 12), nights[i])
                last_hour = None
            hour = f['metrics'].get('date', '')[11:13]
            if hour and hour != last_hour and xs[i] - last_x > 34:    # label each new hour if there is room
                p.setPen(QColor(120, 120, 128))
                p.drawText(QPointF(xs[i] - 10, r.bottom() + 16), f'{hour}h')
                last_hour, last_x = hour, xs[i]

        # axes + y labels
        p.setPen(QPen(QColor(90, 90, 96), 1))
        p.drawRect(r)
        for frac in (0, 0.5, 1):
            v = y0 + frac * (y1 - y0)
            y = r.bottom() - frac * r.height()
            p.setPen(QColor(140, 140, 148))
            p.drawText(QRectF(0, y - 8, self.ML - 6, 16), Qt.AlignRight | Qt.AlignVCenter, fmt.format(v))
        p.save()
        p.translate(12, r.center().y())
        p.rotate(-90)
        p.setPen(QColor(170, 170, 178))
        p.drawText(QRectF(-r.height() / 2, -10, r.height(), 14), Qt.AlignCenter, label)
        p.restore()

        # thresholds: shade the bad side (reject zone red, suspect zone yellow), then dashed lines
        if lims:
            yr, ys = ymap(lims[0]), ymap(lims[1])
            if low_bad:     # small values are bad: zones sit at the bottom
                p.fillRect(QRectF(r.left(), yr, r.width(), r.bottom() - yr), QColor(255, 40, 40, 38))
                p.fillRect(QRectF(r.left(), ys, r.width(), yr - ys), QColor(255, 200, 0, 26))
            else:           # large values are bad: zones sit at the top
                p.fillRect(QRectF(r.left(), r.top(), r.width(), yr - r.top()), QColor(255, 40, 40, 38))
                p.fillRect(QRectF(r.left(), yr, r.width(), ys - yr), QColor(255, 200, 0, 26))
            for v, col in ((lims[0], RING['reject']), (lims[1], RING['suspect'])):
                pen = QPen(col, 1, Qt.DashLine)
                p.setPen(pen)
                p.drawLine(QPointF(r.left(), ymap(v)), QPointF(r.right(), ymap(v)))

        # selection band
        if self._drag:
            a, b = sorted(self._drag)
            p.fillRect(QRectF(a, r.top(), b - a, r.height()), QColor(90, 140, 255, 50))
        # current frame
        cur = [i for i, f in enumerate(self.frames) if f['name'] == self.current]
        if cur:
            p.setPen(QPen(QColor(90, 140, 255), 2))
            p.drawLine(QPointF(xs[cur[0]], r.top()), QPointF(xs[cur[0]], r.bottom()))

        # points
        self._pts = []
        rad = 4.0 if n < 300 else 3.0
        for i, f in enumerate(self.frames):
            v = vals[i]
            off_chart = not np.isfinite(v)
            above = np.isfinite(v) and v > y1          # beyond the display cap
            y = r.bottom() - 2 if off_chart and low_bad else (r.top() + 2 if off_chart else ymap(v))
            if above:
                y = r.top() + rad + 1
            x = xs[i]
            self._pts.append((x, y, f))
            st = self.status.get(f['name'], 'ok')
            fill = filter_color(f['metrics'].get('filter'))
            # status ring sits outside the dot with a dark gap, so it stays visible on any filter colour
            ring = QColor(120, 170, 255) if f['name'] in self.selected else RING.get(st)
            if ring is not None:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(ring, 2.4 if f['name'] in self.selected else 2.0))
                p.drawEllipse(QPointF(x, y), rad + 2.6, rad + 2.6)
            p.setBrush(fill)
            p.setPen(QPen(QColor(15, 15, 18), 1.2))
            if above:        # pinned at the cap: draw a small up-triangle instead of a dot
                p.drawPolygon([QPointF(x, y - rad - 1), QPointF(x - rad, y + rad - 1), QPointF(x + rad, y + rad - 1)])
            else:
                p.drawEllipse(QPointF(x, y), rad, rad)

    # -- interaction -------------------------------------------------------------------------
    def _nearest(self, pos, maxd=10):
        best, bd = None, maxd
        for x, y, f in self._pts:
            d = abs(x - pos.x()) + abs(y - pos.y()) * 0.5
            if d < bd:
                best, bd = f, d
        return best

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = (e.position().x(), e.position().x())

    def mouseMoveEvent(self, e):
        if self._drag:
            self._drag = (self._drag[0], e.position().x())
            self.update()
            return
        f = self._nearest(e.position())
        if f:
            label, get, _, _, fmt = METRICS[self.metric_key]
            v = get(f)
            m = f['metrics']
            QToolTip.showText(e.globalPosition().toPoint(),
                              f'{m.get("date", "")[5:16].replace("T", " ")}  {m.get("filter", "")}  '
                              f'{label}: {fmt.format(v) if np.isfinite(v) else "-"}\n'
                              f'{self.status.get(f["name"], "ok")}', self)
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, e):
        if not self._drag:
            return
        a, b = sorted(self._drag)
        self._drag = None
        if b - a < 4:                       # a click
            f = self._nearest(e.position(), maxd=14)
            if f:
                self.frame_clicked.emit(f['name'])
            elif self.selected:
                self.clear_selection()
            self.update()
            return
        self.selected = {f['name'] for x, _, f in self._pts if a <= x <= b}
        self.selection_changed.emit(sorted(self.selected))
        self.update()
