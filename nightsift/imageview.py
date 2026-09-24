"""Zoomable / pannable image view for the blink panel.

Coordinates are always full-resolution sensor pixels. The cached preview (binned N x N)
is shown scaled up by N; when you zoom in far enough that preview pixels would be
magnified, the full-resolution frame is loaded in the background and swapped in.
"""
import threading
from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from .fitsio import read_fits
from .metrics import _bin

FULLRES_CACHE = 4   # full-res 8-bit frames kept in memory (~47 MB each for 47 MP)


def to_qimage(img8):
    h, w = img8.shape
    img8 = np.ascontiguousarray(img8)
    return QImage(img8.data, w, h, w, QImage.Format_Grayscale8).copy()


class _Loader(QObject):
    loaded = Signal(str, object)   # key, uint16 full-res array


class ImageView(QGraphicsView):
    zoom_changed = Signal(float)   # screen pixels per sensor pixel

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(12, 12, 14))
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.item)
        self.border = self.scene().addRect(QRectF(), QPen(Qt.NoPen))
        self.border.setZValue(2)
        self._fit = True               # fit-to-window mode (re-fits on resize / new frame)
        self._full = OrderedDict()     # key -> uint16 full-res
        self._loading = set()
        self._key, self._path, self._lut, self._shape = None, None, None, None
        self._showing_full = False
        self._flip, self._offset = False, (0.0, 0.0)   # display-only alignment (compare mode)
        self.osc = False               # colour data: show the full-res frame as 2x2 superpixels
        self._loader = _Loader()
        self._loader.loaded.connect(self._full_loaded)
        self.placeholder = self.scene().addText('')
        self.placeholder.setDefaultTextColor(QColor(140, 140, 140))

    # -- content ---------------------------------------------------------------------------
    def set_message(self, text):
        self.item.setPixmap(QPixmap())
        self.border.setPen(QPen(Qt.NoPen))
        self._key = None
        self.placeholder.setPlainText(text)
        self.scene().setSceneRect(self.placeholder.boundingRect())

    def set_frame(self, key, path, preview, lut, full_shape, border=None, flip=False, offset=(0.0, 0.0)):
        """Show a frame. preview: binned uint16; lut: 65536-entry uint8 table; full_shape: (h, w).
        flip / offset (sensor pixels): display the frame rotated 180° and shifted, to line it up with
        another frame (meridian flip + dither) while blinking."""
        self.placeholder.setPlainText('')
        self._flip, self._offset = bool(flip), (float(offset[0]), float(offset[1]))
        new_geometry = self._shape != tuple(full_shape)
        self._key, self._path, self._lut, self._shape = key, path, lut, tuple(full_shape)
        self._preview_bin = full_shape[1] / preview.shape[1]
        self.scene().setSceneRect(QRectF(0, 0, full_shape[1], full_shape[0]))
        full = self._full.get(key)
        if full is not None and self._wants_full():
            self._show(lut[full], 1.0)
        else:
            self._show(lut[preview], self._preview_bin)
            self._maybe_load_full()
        pen = QPen(border, 6) if border is not None else QPen(Qt.NoPen)
        pen.setCosmetic(True)   # constant 6 screen pixels at any zoom
        self.border.setPen(pen)
        self.border.setRect(QRectF(0, 0, full_shape[1], full_shape[0]))
        if self._fit or new_geometry:
            self.fit()

    def _show(self, img8, scale):
        if self._flip:
            img8 = img8[::-1, ::-1]
        self.item.setPixmap(QPixmap.fromImage(to_qimage(img8)))
        self.item.setScale(scale)
        self.item.setPos(*self._offset)
        self._showing_full = scale == 1.0

    # -- zoom ------------------------------------------------------------------------------
    def current_zoom(self):
        return self.transform().m11()

    def _wants_full(self):
        """Full-res is worth loading once preview pixels would be magnified on screen."""
        return self._shape is not None and self.current_zoom() * getattr(self, '_preview_bin', 1) > 1.05

    def fit(self):
        self._fit = True
        if self._shape:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self.zoom_changed.emit(self.current_zoom())

    def set_zoom(self, z, anchor=None):
        z = float(np.clip(z, 0.02, 16))
        if anchor is not None:
            self.setTransformationAnchor(QGraphicsView.NoAnchor)
            before = self.mapToScene(anchor)
        self.resetTransform()
        self.scale(z, z)
        if anchor is not None:
            after = self.mapToScene(anchor)
            d = after - before
            self.translate(d.x(), d.y())
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._fit = False
        self.zoom_changed.emit(z)
        self._maybe_load_full()

    def zoom_by(self, factor, anchor=None):
        self.set_zoom(self.current_zoom() * factor, anchor)

    def one_to_one(self):
        c = self.mapToScene(self.viewport().rect().center())
        self.set_zoom(1.0)
        self.centerOn(c)

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120
        if steps:
            self.zoom_by(1.25 ** steps, e.position().toPoint())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()

    # -- full resolution -------------------------------------------------------------------
    def _maybe_load_full(self):
        if not self._wants_full() or self._key is None:
            return
        key, path = self._key, self._path
        if key in self._full:
            if not self._showing_full:
                self._show(self._lut[self._full[key]], 1.0)
            return
        if key in self._loading:
            return
        self._loading.add(key)

        def work():
            try:
                _, img = read_fits(path)
                if self.osc:
                    img = _bin(img, 2, np.uint16 if img.dtype == np.uint16 else np.float32)
                if img.dtype != np.uint16:
                    img = np.clip(img, 0, 65535).astype(np.uint16)
            except Exception:
                img = None
            self._loader.loaded.emit(key, img)
        threading.Thread(target=work, daemon=True).start()

    def _full_loaded(self, key, img):
        self._loading.discard(key)
        if img is None:
            return
        self._full[key] = img
        while len(self._full) > FULLRES_CACHE:
            self._full.popitem(last=False)
        if key == self._key and self._wants_full():
            self._show(self._lut[img], 1.0)

    def restretch(self, lut):
        """Re-render the current frame with a different stretch (same zoom/position)."""
        self._lut = lut
        if self._key is None:
            return
        if self._showing_full and self._key in self._full:
            self._show(lut[self._full[self._key]], 1.0)
