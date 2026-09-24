"""nightsift desktop app (PySide6).

Launch with:  python -m nightsift.app
(the tiny launcher keeps Qt out of the scan worker processes)
"""
import os
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QSettings, QSortFilterProxyModel, Qt,
                            QThread, QTimer, Signal)
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QSlider, QSpinBox, QSplitter, QStyleFactory, QTableView,
                               QTableWidget, QTableWidgetItem, QTabBar, QToolButton, QVBoxLayout, QWidget)

from .engine import DEFAULT_WORKERS, build_frames, scan_iter, write_candidates
from .scoring import night_of
from . import movedialog
from .align import align
from .imageview import ImageView
from .scoring import _limits
from .timeline import METRICS, Timeline
from .store import Store
from .viewer import stf_lut

LINEAR_LUT = (np.arange(65536) >> 8).astype(np.uint8)   # autostretch off: raw linear data
BORDER = {'reject': QColor(230, 50, 50), 'suspect': QColor(240, 190, 30), 'reference': QColor(70, 150, 255)}

APP = 'NightSift'
VIEWS = [('all', 'All'), ('flagged', 'Flagged'), ('suspect', 'Suspect'), ('kept', 'Kept'), ('rejected', 'Rejected'),
         ('mine', 'My choices')]
TIER_BG = {'reject': QColor(120, 30, 30), 'suspect': QColor(110, 90, 20)}
MIN_S, MAX_S, DEFAULT_S = 0.5, 2.0, 1.0


# ---------------------------------------------------------------------------------------------
# background scan
# ---------------------------------------------------------------------------------------------
class OpenThread(QThread):
    """Opens a project off the UI thread: a sleeping hard disk can take seconds to answer."""
    opened = Signal(object)
    failed = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.opened.emit(Store(self.path))
        except OSError as e:
            self.failed.emit(str(e))


class ScanThread(QThread):
    progress = Signal(int, int, str, str)   # done, total, path, error ('' if none)
    phase = Signal(str, object, object)     # see engine.scan_iter(on_phase=...)
    finished_scan = Signal(int, float)      # frames measured, seconds

    def __init__(self, store, workers=DEFAULT_WORKERS, force=False):
        super().__init__()
        self.store, self.workers, self._cancel, self.force = store, workers, False, force

    def cancel(self):
        self._cancel = True

    def run(self):
        t0, n = time.perf_counter(), 0
        for done, total, path, err in scan_iter(self.store, self.workers, cancelled=lambda: self._cancel,
                                                force=self.force, on_phase=self.phase.emit):
            n = done
            self.progress.emit(done, total, path, err or '')
        self.finished_scan.emit(n, time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------------
# frame table
# ---------------------------------------------------------------------------------------------
def _fmt_pct(v):
    return f'{v:.0%}' if np.isfinite(v) else '-'


def _duration(secs):
    secs = int(round(secs))
    if secs < 90:
        return f'{secs} s'
    if secs < 3600:
        return f'{secs // 60} min {secs % 60:02d} s'
    return f'{secs // 3600} h {secs % 3600 // 60:02d} min'


def why_text(f):
    """Reasons for flagged frames; for OK frames, how they compare with their group anyway."""
    if f['reasons']:
        return ', '.join(f['reasons'])
    return f'stars {_fmt_pct(f["r_stars"])}, HFR {_fmt_pct(f["r_hfr"])}'


class FrameModel(QAbstractTableModel):
    COLS = ['Folder', 'Filter', 'Time', 'Status', 'Why', 'Stars', 'Stars %', 'HFR', 'HFR %', 'FWHM″', 'Ecc',
            'SNR', 'Weight', 'Sky ×', 'File']
    NUMERIC = {'Stars', 'Stars %', 'HFR', 'HFR %', 'FWHM″', 'Ecc', 'SNR', 'Weight', 'Sky ×'}
    TIPS = {'Stars %': 'Star count vs the median frame of the same night + filter',
            'HFR': 'Half-flux radius in pixels (NINA definition)',
            'HFR %': 'HFR vs the median frame of the same night + filter (>100% = bigger stars)',
            'FWHM″': 'Star FWHM in arcseconds (absolute: compare nights with this)',
            'Ecc': 'Eccentricity, PixInsight scale: 0 = round, >0.55 noticeably elongated',
            'SNR': 'SNR of a typical bright star: falls with clouds, bright sky and blur',
            'Weight': '(SNR vs group median)² — roughly this frame\'s share of signal in the stack',
            'Sky ×': 'Sky background vs the median frame of the same night + filter'}

    def __init__(self):
        super().__init__()
        self.frames, self.store = [], None

    def set_frames(self, frames, store):
        self.beginResetModel()
        self.frames, self.store = frames, store
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frames)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return self.COLS[section]
            if role == Qt.ToolTipRole:
                return self.TIPS.get(self.COLS[section])
        return None

    def status(self, f):
        d = self.store.decisions.get(f['name'])
        if d:
            return f'{d} (you)'
        return f['tier']

    def data(self, index, role=Qt.DisplayRole):
        f = self.frames[index.row()]
        m, c = f['metrics'], self.COLS[index.column()]
        if role in (Qt.DisplayRole, Qt.UserRole):
            sort = role == Qt.UserRole
            if c == 'Folder':
                return f['folder']
            if c == 'Filter':
                return m.get('filter', '')
            if c == 'Time':
                return m.get('date', '') if sort else m.get('date', '')[5:16].replace('T', ' ')
            if c == 'Status':
                return {'reject': 0, 'suspect': 1, 'ok': 2}.get(f['tier'], 3) if sort else self.status(f)
            if c == 'Why':
                return why_text(f)
            if c == 'File':
                return f['name']
            v = {'Stars': m.get('nstars', 0), 'Stars %': f['r_stars'], 'HFR': m.get('hfr', np.nan),
                 'HFR %': f['r_hfr'], 'FWHM″': m.get('fwhm', np.nan) * m.get('scale', np.nan),
                 'Ecc': m.get('ecc', np.nan), 'SNR': m.get('snr', np.nan), 'Weight': f['weight'],
                 'Sky ×': f['r_sky']}[c]
            v = np.nan if v is None else v
            if sort:
                return float(v) if np.isfinite(v) else -1.0
            if not np.isfinite(v):
                return '-'
            if c in ('Stars %', 'HFR %', 'Weight'):
                return _fmt_pct(v)
            if c in ('HFR', 'Ecc', 'Sky ×'):
                return f'{v:.2f}'
            if c == 'FWHM″':
                return f'{v:.1f}'
            return f'{v:.0f}'
        if role == Qt.BackgroundRole:
            if self.store.is_rejected(f):
                return TIER_BG['reject']
            if f['tier'] == 'suspect':
                return TIER_BG['suspect']
        if role == Qt.ForegroundRole and c == 'Why' and not f['reasons']:
            return QColor(150, 150, 150)   # OK frames: plain info, not a warning
        if role == Qt.TextAlignmentRole and c in self.NUMERIC:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ToolTipRole:
            return f['path']
        return None


class FrameFilter(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.view, self.group = 'all', None
        self.setSortRole(Qt.UserRole)

    def set_view(self, v):
        self.view = v
        self.invalidateFilter()

    def set_group(self, g):
        self.group = g
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        src = self.sourceModel()
        f = src.frames[row]
        if self.group and f['group'] != self.group:
            return False
        rej = src.store.is_rejected(f)
        return {'all': True, 'flagged': f['tier'] != 'ok' or rej, 'suspect': f['tier'] == 'suspect',
                'kept': not rej, 'rejected': rej, 'mine': f['name'] in src.store.decisions}[self.view]


# ---------------------------------------------------------------------------------------------
# simple preview (full blink viewer comes in step 2)
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, initial=None):
        super().__init__()
        self.setWindowTitle(APP)
        self.settings = QSettings(APP, APP)
        self.store, self.frames, self.stats, self.scan_thread = None, [], {}, None
        self._paths = {}          # basename -> current path; the list never re-walks the disk itself
        self._open_thread = None
        self._prev_cache, self._lut_cache = {}, {}
        self._align_cache = {}    # (frame, reference) -> (flip, dx, dy, score) in preview pixels
        self._scan_t0 = 0.0
        self._build_ui()
        self._set_enabled(False)
        self.refresh_timer = QTimer(self, interval=1500, timeout=self.refresh)
        start = initial or self.settings.value('last_project')
        if start and os.path.isdir(start):
            QTimer.singleShot(0, lambda: self.open_project(start))

    # -- layout -------------------------------------------------------------------------------
    def _build_ui(self):
        # row 1: project + reject folder
        self.btn_open = QPushButton('Open project…')
        self.btn_open.clicked.connect(self.choose_project)
        self.lbl_project = QLabel('<i>no project</i>')
        self.lbl_project.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_reject = QLabel('')
        self.lbl_reject.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.btn_reject = QPushButton('Change…')
        self.btn_reject.setToolTip('Folder where rejected frames are moved (outside the project)')
        self.btn_reject.clicked.connect(self.choose_reject_dir)
        r1 = QHBoxLayout()
        r1.addWidget(self.btn_open)
        r1.addWidget(self.lbl_project, 1)
        r1.addWidget(QLabel('Rejects →'))
        r1.addWidget(self.lbl_reject, 1)
        r1.addWidget(self.btn_reject)

        # row 2: strictness, OSC, scan
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(MIN_S * 100), int(MAX_S * 100))
        self.slider.setSingleStep(5)
        self.slider.setPageStep(10)
        self.slider.setFixedWidth(220)
        self.slider.setToolTip('How strict stage 1 is. 1.0 = default, lower = lenient, higher = strict')
        self.spin = QDoubleSpinBox()
        self.spin.setRange(MIN_S, MAX_S)
        self.spin.setSingleStep(0.05)
        self.spin.setDecimals(2)
        self.btn_reset = QPushButton('Reset')
        self.btn_reset.setToolTip(f'Back to the default strictness ({DEFAULT_S})')
        self.slider.valueChanged.connect(lambda v: self.spin.setValue(v / 100))
        self.spin.valueChanged.connect(self._strictness_changed)
        self.btn_reset.clicked.connect(lambda: self.spin.setValue(DEFAULT_S))
        self.chk_osc = QCheckBox('Colour camera (OSC)')
        self.chk_osc.setToolTip('Tick for one-shot-colour (Bayer) data. Frames are measured on 2x2 '
                                'superpixels so the colour pattern does not distort star shapes.')
        self.chk_osc.clicked.connect(self._osc_clicked)
        self.btn_scan = QPushButton('Scan')
        self.btn_scan.setToolTip('Measure new / changed frames (already measured frames are cached)')
        self.btn_scan.clicked.connect(self.start_scan)
        self.btn_rescan = QPushButton('Rescan all…')
        self.btn_rescan.setToolTip('Re-measure every frame and rebuild all previews. '
                                   'Keeps your reject/keep choices, moved files and settings.')
        self.btn_rescan.clicked.connect(self.rescan_all)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.clicked.connect(self.stop_scan)
        self.btn_stop.setEnabled(False)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel('Strictness'))
        r2.addWidget(self.slider)
        r2.addWidget(self.spin)
        r2.addWidget(self.btn_reset)
        r2.addSpacing(20)
        r2.addWidget(self.chk_osc)
        r2.addStretch(1)
        r2.addWidget(self.btn_scan)
        r2.addWidget(self.btn_rescan)
        r2.addWidget(self.btn_stop)
        r2.addSpacing(20)
        self.btn_move = QPushButton('Move rejects…')
        self.btn_move.setToolTip('Preview, then move rejected frames out of the project (never deletes)')
        self.btn_move.clicked.connect(self.move_rejects)
        self.btn_restore = QPushButton('Restore…')
        self.btn_restore.setToolTip('Preview, then move frames back from the reject folder')
        self.btn_restore.clicked.connect(self.restore_rejects)
        r2.addWidget(self.btn_move)
        r2.addWidget(self.btn_restore)

        # left: groups + frames
        self.groups = QTableWidget(0, 7)
        self.groups.setHorizontalHeaderLabels(['Night', 'Folder', 'Filter', 'Frames', 'Reject', 'Suspect', 'OK'])
        self.groups.verticalHeader().hide()
        self.groups.verticalHeader().setDefaultSectionSize(22)
        self.groups.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.groups.setSelectionMode(QAbstractItemView.SingleSelection)
        self.groups.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.groups.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.groups.horizontalHeader().setStretchLastSection(True)
        self.groups.itemSelectionChanged.connect(self._group_selected)
        self.groups.setToolTip('Click a row to show only that night + filter; click "All groups" to reset')

        self.tabs = QTabBar()
        for _, label in VIEWS:
            self.tabs.addTab(label)
        self.tabs.setTabToolTip(len(VIEWS) - 1, 'Only frames where you overruled NightSift (X / Reject / Keep '
                                               'selected). U undoes a choice.')
        self.tabs.currentChanged.connect(self._tab_changed)
        self.btn_allgroups = QPushButton('All groups')
        self.btn_allgroups.clicked.connect(lambda: self.groups.clearSelection())
        self.btn_clear_mine = QPushButton('Clear all my choices…')
        self.btn_clear_mine.setToolTip('Forget every manual reject / keep in this project: all frames go back '
                                       'to the automatic result')
        self.btn_clear_mine.clicked.connect(self.clear_my_choices)
        self.btn_clear_mine.hide()
        tabrow = QHBoxLayout()
        tabrow.addWidget(self.tabs)
        tabrow.addStretch(1)
        tabrow.addWidget(self.btn_clear_mine)
        tabrow.addWidget(self.btn_allgroups)

        self.model = FrameModel()
        self.proxy = FrameFilter()
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(2, Qt.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().currentRowChanged.connect(self._row_changed)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.groups, 1)
        lv.addLayout(tabrow)
        lv.addWidget(self.table, 4)

        self.view = ImageView()
        self.view.setFocusPolicy(Qt.NoFocus)        # keys stay with the frame list
        self.view.set_message('Open a project folder to start')
        self.view.zoom_changed.connect(lambda z: self.lbl_zoom.setText(f'{z * 100:.0f}%'))

        def tb(text, tip, slot, checkable=False):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setFocusPolicy(Qt.NoFocus)
            (b.toggled if checkable else b.clicked).connect(slot)
            return b
        self.btn_prev = tb('◀', 'Previous frame (← or ↑)', lambda: self.step(-1))
        self.btn_play = tb('▶ Play', 'Blink through the frames in the list (Space)', self.set_playing, True)
        self.btn_next = tb('▶', 'Next frame (→ or ↓)', lambda: self.step(1))
        self.fps = QSpinBox()
        self.fps.setRange(1, 20)
        self.fps.setValue(4)
        self.fps.setSuffix(' fps')
        self.fps.setFocusPolicy(Qt.NoFocus)
        self.fps.setToolTip('Blink speed (frames per second)')
        self.fps.valueChanged.connect(lambda v: self.play_timer.setInterval(int(1000 / v)))
        self.btn_reject_frame = tb('✗ Reject / keep  (X)', 'Toggle reject for the current frame', self.toggle_current)
        self.btn_compare = tb('⇄ Compare to best  (R)',
                              'Flip between this frame and a reference frame of the same night + filter, at the '
                              'same zoom, position and stretch. The reference is the best clean frame on the '
                              'metric this frame was flagged for (weight, FWHM, HFR or eccentricity). '
                              'Press R repeatedly to blink them.',
                              self.set_compare, True)
        self.btn_undo_frame = tb('↶ Undo my choice  (U)', 'Forget your reject / keep for this frame and go back '
                                 'to the automatic result', self.undo_current)
        self.btn_stretch = QCheckBox('Autostretch')
        self.btn_stretch.setFocusPolicy(Qt.NoFocus)
        self.btn_stretch.setToolTip('Auto-stretch on/off (A). Off shows the raw linear data')
        self.btn_stretch.setChecked(True)
        self.btn_stretch.toggled.connect(self._stretch_changed)
        self.chk_linked = QCheckBox('Same stretch for all frames of a filter')
        self.chk_linked.setFocusPolicy(Qt.NoFocus)
        self.chk_linked.setToolTip('On: every frame of a night + filter uses one stretch, so clouds and twilight '
                                   'show up as brighter frames while blinking.\n'
                                   'Off: each frame is stretched on its own (all look similar).')
        self.chk_linked.setChecked(True)
        self.chk_linked.toggled.connect(self._stretch_changed)
        self.btn_fit = tb('Fit', 'Fit to window (F)', self.view.fit)
        self.btn_11 = tb('1:1', 'Actual pixels (1)', self.view.one_to_one)
        self.btn_zin = tb('+', 'Zoom in (+ or mouse wheel)', lambda: self.view.zoom_by(1.25))
        self.btn_zout = tb('−', 'Zoom out (- or mouse wheel)', lambda: self.view.zoom_by(0.8))
        self.lbl_zoom = QLabel('')
        self.lbl_zoom.setMinimumWidth(44)
        vt = QHBoxLayout()
        for w in (self.btn_prev, self.btn_play, self.btn_next, self.fps):
            vt.addWidget(w)
        vt.addSpacing(12)
        vt.addWidget(self.btn_reject_frame)
        vt.addWidget(self.btn_undo_frame)
        vt.addWidget(self.btn_compare)
        vt.addStretch(1)
        for w in (self.btn_stretch, self.chk_linked):
            vt.addWidget(w)
        vt.addSpacing(12)
        for w in (self.btn_zout, self.lbl_zoom, self.btn_zin, self.btn_fit, self.btn_11):
            vt.addWidget(w)
        self.lbl_frame = QLabel('')
        self.lbl_frame.setWordWrap(True)
        self.lbl_frame.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addLayout(vt)
        rv.addWidget(self.view, 1)
        rv.addWidget(self.lbl_frame)

        self.play_timer = QTimer(self, interval=250, timeout=lambda: self.step(1, wrap=True))
        for keys, slot in ((('Right',), lambda: self.step(1)), (('Left',), lambda: self.step(-1)),
                           (('Space',), lambda: self.btn_play.toggle()),
                           (('X', 'Delete'), self.toggle_current), (('U',), self.undo_current),
                           (('R',), lambda: self.btn_compare.toggle()),
                           (('A',), lambda: self.btn_stretch.toggle()),
                           (('F',), self.view.fit), (('1',), self.view.one_to_one),
                           (('+', '='), lambda: self.view.zoom_by(1.25)), (('-',), lambda: self.view.zoom_by(0.8))):
            for k in keys:
                QShortcut(QKeySequence(k), self, activated=slot, context=Qt.WindowShortcut)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([760, 740])

        # timeline
        self.timeline = Timeline()
        self.timeline.frame_clicked.connect(self._timeline_clicked)
        self.timeline.selection_changed.connect(self._timeline_selected)
        self.cmb_metric = QComboBox()
        for key, (label, *_rest) in METRICS.items():
            self.cmb_metric.addItem(label, key)
        self.cmb_metric.setFocusPolicy(Qt.NoFocus)
        self.cmb_metric.currentIndexChanged.connect(
            lambda i: self.timeline.set_metric(self.cmb_metric.itemData(i)))
        self.lbl_sel = QLabel('Drag across the chart to select a range of frames')
        self.lbl_sel.setStyleSheet('color:#999')
        self.btn_sel_reject = QPushButton('Reject selected')
        self.btn_sel_keep = QPushButton('Keep selected')
        self.btn_sel_clear = QPushButton('Clear selection')
        for b, fn in ((self.btn_sel_reject, lambda: self._bulk(True)), (self.btn_sel_keep, lambda: self._bulk(False)),
                      (self.btn_sel_clear, self.timeline.clear_selection)):
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(fn)
            b.hide()
        th = QHBoxLayout()
        th.addWidget(QLabel('Timeline'))
        th.addWidget(self.cmb_metric)
        th.addSpacing(16)
        th.addWidget(self.lbl_sel)
        for b in (self.btn_sel_reject, self.btn_sel_keep, self.btn_sel_clear):
            th.addWidget(b)
        th.addStretch(1)
        bottom = QWidget()
        bv = QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addLayout(th)
        bv.addWidget(self.timeline, 1)
        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(split)
        vsplit.addWidget(bottom)
        vsplit.setSizes([640, 230])

        central = QWidget()
        v = QVBoxLayout(central)
        v.addLayout(r1)
        v.addLayout(r2)
        v.addWidget(vsplit, 1)
        self.setCentralWidget(central)

        # status bar
        self.progress = QProgressBar()
        self.progress.setFixedWidth(260)
        self.progress.hide()
        self.lbl_status = QLabel('')
        self.lbl_counts = QLabel('')
        self.statusBar().addWidget(self.lbl_status, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.lbl_counts)

        act = QAction('Open project', self, shortcut=QKeySequence.Open, triggered=self.choose_project)
        self.addAction(act)
        self.resize(1500, 900)

    def _set_enabled(self, on):
        for w in (self.slider, self.spin, self.btn_reset, self.chk_osc, self.btn_scan, self.btn_reject,
                  self.btn_move, self.btn_restore, self.btn_rescan):
            w.setEnabled(on)

    # -- project ------------------------------------------------------------------------------
    def choose_project(self):
        start = self.store.root if self.store else (self.settings.value('last_project') or '')
        d = QFileDialog.getExistingDirectory(self, 'Open project folder (searched recursively)', start)
        if d:
            self.open_project(d)

    def open_project(self, path):
        if self.scan_thread:
            self.stop_scan()
            self.scan_thread.wait()
        if self._open_thread and self._open_thread.isRunning():
            return
        self._set_enabled(False)
        self.btn_open.setEnabled(False)
        self.progress.setRange(0, 0)          # busy indicator
        self.progress.show()
        self.lbl_status.setText(f'Opening {path} … (a hard disk may take a few seconds to spin up)')
        self._open_thread = OpenThread(path)
        self._open_thread.opened.connect(self._project_opened)
        self._open_thread.failed.connect(lambda e: self._open_failed(path, e))
        self._open_thread.start()

    def _open_failed(self, path, err):
        self.progress.hide()
        self.btn_open.setEnabled(True)
        self._set_enabled(self.store is not None)
        self.lbl_status.setText('')
        QMessageBox.warning(self, APP, f'Cannot open {path}:\n{err}')

    def _cached_paths(self):
        """Where frames were last seen, from the cache alone (no disk access)."""
        paths = {}
        with self.store.lock:
            for n, m in self.store.metrics.items():
                paths[n] = self.store.rejected_path(n) if n in self.store.moved else m['file']
        return paths

    def _project_opened(self, store):
        self.btn_open.setEnabled(True)
        self.store = store
        self._paths = self._cached_paths()
        self.settings.setValue('last_project', self.store.root)
        self._prev_cache.clear()
        self._lut_cache.clear()
        self.setWindowTitle(f'{APP} — {self.store.root}')
        self.lbl_project.setText(self.store.root)
        self.lbl_reject.setText(self.store.reject_dir)
        self.spin.blockSignals(True)
        self.spin.setValue(self.store.strictness)
        self.slider.setValue(int(round(self.store.strictness * 100)))
        self.spin.blockSignals(False)
        self._set_enabled(True)
        self.refresh(keep_selection=False)
        self._fit_columns()
        self.chk_osc.setChecked(self.store.osc)   # first scan of a project decides it from the headers
        self.start_scan()

    def choose_reject_dir(self):
        if self.store.moved:
            QMessageBox.information(self, APP, f'{len(self.store.moved)} frame(s) are currently in\n'
                                    f'{self.store.reject_dir}\n\nRestore them before changing the reject folder.')
            return
        d = QFileDialog.getExistingDirectory(self, 'Folder for rejected frames (outside the project)',
                                             os.path.dirname(self.store.reject_dir))
        if not d:
            return
        d = os.path.abspath(d)
        if os.path.commonpath([d, self.store.root]) == self.store.root:
            QMessageBox.warning(self, APP, 'The reject folder must be outside the project folder, '
                                'otherwise WBPP would pick the rejects up again.')
            return
        self.store.reject_dir = self.store.config['reject_dir'] = d
        self.store.save('config')
        self.lbl_reject.setText(d)

    # -- settings -----------------------------------------------------------------------------
    def _strictness_changed(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * 100)))
        self.slider.blockSignals(False)
        if self.store:
            self.store.config['strictness'] = round(v, 2)
            self.store.save('config')
            self.refresh()

    def _osc_clicked(self, checked):
        n = len(self.store.metrics)
        if n and QMessageBox.question(
                self, APP, f'Switch to {"colour (OSC)" if checked else "mono"} mode?\n\n'
                f'All {n} frames will be measured again.') != QMessageBox.Yes:
            self.chk_osc.setChecked(not checked)
            return
        self.store.config['osc'] = checked
        self.store.save('config')
        self._prev_cache.clear()
        self.start_scan()

    # -- scanning -----------------------------------------------------------------------------
    def rescan_all(self):
        if not self.store or (self.scan_thread and self.scan_thread.isRunning()):
            return
        n = len(self.frames)
        if QMessageBox.question(
                self, APP, f'Re-measure all {n} frames and rebuild their previews?\n\n'
                'Your reject / keep choices, moved files, reject folder and settings are kept.\n'
                'Current results stay visible and are replaced frame by frame; Stop at any time.\n\n'
                'This reads every file again, so it takes as long as the first scan '
                '(about 1.5 frames/s from a hard disk, several per second from an SSD).') != QMessageBox.Yes:
            return
        self._prev_cache.clear()
        self._lut_cache.clear()
        self.view._full.clear()
        self.start_scan(force=True)

    def start_scan(self, force=False):
        if not self.store or (self.scan_thread and self.scan_thread.isRunning()):
            return
        self.scan_thread = ScanThread(self.store, force=force)
        self._scan_start = time.perf_counter()
        self.scan_thread.progress.connect(self._scan_progress)
        self.scan_thread.phase.connect(self._scan_phase)
        self.scan_thread.finished_scan.connect(self._scan_done)
        self._scan_t0 = time.perf_counter()
        self._scan_errors = 0
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        for w in (self.chk_osc, self.btn_move, self.btn_restore, self.btn_reject, self.btn_rescan):
            w.setEnabled(False)
        self.progress.setRange(0, 0)          # busy until the first phase reports
        self.progress.show()
        self.lbl_status.setText('Finding files… (a hard disk may take a few seconds to spin up)')
        self.scan_thread.start()
        self.refresh_timer.start()

    def _scan_phase(self, phase, a, b):
        if phase == 'files':
            self._paths = {os.path.basename(p): p for p in a}
            self.lbl_status.setText(f'Found {len(a)} FITS files. Checking which are light frames…')
        elif phase == 'check':
            self.progress.setRange(0, b)
            self.progress.setValue(a)
            self.lbl_status.setText(f'Checking file headers {a}/{b} (skipping flats, darks, bias)…')
        elif phase == 'osc':
            self.chk_osc.setChecked(bool(a))
        elif phase == 'measure':
            self._scan_t0 = time.perf_counter()
            self.progress.setRange(0, b)
            self.progress.setValue(0)
            self.lbl_status.setText(f'Measuring {b} frame(s)… (first results in a few seconds)')

    def stop_scan(self):
        if self.scan_thread:
            self.scan_thread.cancel()
            self.lbl_status.setText('Stopping after the frames in progress…')

    def _scan_progress(self, done, total, path, err):
        if err:
            self._scan_errors += 1
        self.progress.show()
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        el = time.perf_counter() - self._scan_t0
        rate = done / el if el > 0 else 0
        eta = (total - done) / rate if rate else 0
        errs = f' · {self._scan_errors} unreadable' if self._scan_errors else ''
        mins = f'{eta / 60:.0f} min' if eta >= 90 else f'{eta:.0f} s'
        self._update_counts()
        if self.scan_thread and self.scan_thread._cancel:
            self.lbl_status.setText(f'Stopping… finishing the frames already being read ({done}/{total} done)')
        else:
            self.lbl_status.setText(f'Measuring {done}/{total} · {rate:.1f} frames/s · about {mins} left{errs}')

    def _scan_done(self, n, secs):
        if n:   # remember the last scan that measured something (shown in the status bar)
            self.store.config['last_scan'] = dict(frames=n, seconds=round(secs, 1),
                                                  rescan=bool(self.scan_thread and self.scan_thread.force),
                                                  when=time.strftime('%Y-%m-%d %H:%M'))
            self.store.save('config')
        self.refresh_timer.stop()
        self.progress.hide()
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        for w in (self.chk_osc, self.btn_move, self.btn_restore, self.btn_reject, self.btn_rescan):
            w.setEnabled(True)
        self._prev_cache.clear()             # previews may have been rebuilt
        self._lut_cache.clear()
        self.refresh()
        if n:
            self._fit_columns()
        write_candidates(self.store, self.frames)
        errs = f', {self._scan_errors} unreadable' if self._scan_errors else ''
        bayer = any(f['metrics'].get('bayer') for f in self.frames)
        hint = '   ⚠ headers say these are colour (Bayer) frames — tick "Colour camera (OSC)"' \
            if bayer and not self.store.osc else ''
        what = 'Rescanned' if self.scan_thread and self.scan_thread.force else 'Scanned'
        new = '' if what == 'Rescanned' else 'new '
        self.lbl_status.setText((f'{what} {n} {new}frame(s) in {secs:.0f} s{errs}.' if n else
                                 'Up to date — all frames already measured.') + hint)
        self.scan_thread = None

    # -- data refresh -------------------------------------------------------------------------
    def refresh(self, keep_selection=True):
        if not self.store:
            return
        cur = self._current_frame() if keep_selection else None
        self.frames, self.stats = build_frames(self.store, self.spin.value(), self._paths)
        for f in self.frames:
            orig = self.store.moved.get(f['name'], f['path'])
            f['folder'] = self.store.bucket(orig)
            s, m = self.stats[f['group']], f['metrics']
            f['r_stars'] = m.get('nstars', 0) / s['nstars'] if s['nstars'] else np.nan
            f['r_hfr'] = m.get('hfr', np.nan) / s['hfr'] if s['hfr'] else np.nan
            f['r_sky'] = m.get('bg', np.nan) / s['bg'] if s['bg'] else np.nan
            f['r_fwhm'] = m.get('fwhm', np.nan) / s['fwhm'] if s.get('fwhm') else np.nan
            r_snr = m.get('snr', np.nan) / s['snr'] if s.get('snr') else np.nan
            f['weight'] = r_snr ** 2
        self.model.set_frames(self.frames, self.store)
        self._fill_groups()
        self._update_counts()
        self._update_timeline()
        if cur:
            self._select_name(cur['name'])

    def _fill_groups(self):
        sel = self.proxy.group
        groups = {}
        for f in self.frames:
            groups.setdefault(f['group'], []).append(f)
        self.groups.blockSignals(True)
        self.groups.setRowCount(len(groups))
        for r, (g, fs) in enumerate(groups.items()):
            m0 = fs[0]['metrics']
            folders = sorted({f['folder'] for f in fs})
            nrej = sum(self.store.is_rejected(f) for f in fs)
            nsus = sum(f['tier'] == 'suspect' and not self.store.is_rejected(f) for f in fs)
            vals = [night_of(m0), ', '.join(folders), f'{m0.get("filter", "")} {m0.get("exptime", 0):g}s',
                    len(fs), nrej, nsus, len(fs) - nrej - nsus]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setData(Qt.UserRole, g)
                if c >= 3:
                    it.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                if c == 4 and nrej:
                    it.setBackground(TIER_BG['reject'])
                if c == 5 and nsus:
                    it.setBackground(TIER_BG['suspect'])
                self.groups.setItem(r, c, it)
            if g == sel:
                self.groups.selectRow(r)
        self.groups.blockSignals(False)

    def _group_selected(self):
        items = self.groups.selectedItems()
        self.proxy.set_group(items[0].data(Qt.UserRole) if items else None)
        self._update_counts()
        self.timeline.clear_selection()
        self._update_timeline()

    def _fit_columns(self):
        """Size columns to their contents (capped so the Why column cannot take over)."""
        self.table.resizeColumnsToContents()
        for c in range(self.model.columnCount()):
            self.table.setColumnWidth(c, min(self.table.columnWidth(c), 340))

    # -- timeline -------------------------------------------------------------------------------
    def _status(self, f):
        if self.store.is_rejected(f):
            return 'reject'
        return 'suspect' if f['tier'] == 'suspect' else 'ok'

    def _update_timeline(self):
        if not self.store:
            return
        g = self.proxy.group
        fs = [f for f in self.frames if not g or f['group'] == g]
        cur = self._current_frame()
        self.timeline.set_data(fs, {f['name']: self._status(f) for f in fs}, _limits(self.spin.value()),
                               cur['name'] if cur else None)

    def _timeline_clicked(self, name):
        self._select_name(name)
        if not self._current_frame() or self._current_frame()['name'] != name:
            self.tabs.setCurrentIndex(0)          # frame hidden by the current tab: show All
            self._select_name(name)

    def _timeline_selected(self, names):
        on = bool(names)
        self.lbl_sel.setText(f'{len(names)} frame(s) selected:' if on else
                             'Drag across the chart to select a range of frames')
        for b in (self.btn_sel_reject, self.btn_sel_keep, self.btn_sel_clear):
            b.setVisible(on)

    def _set_decision(self, f, want_reject):
        """Record a manual choice; matching what stage 1 decided removes the override."""
        if want_reject == (f['tier'] == 'reject'):
            self.store.decisions.pop(f['name'], None)
        else:
            self.store.decisions[f['name']] = 'reject' if want_reject else 'keep'

    def _bulk(self, reject):
        names = set(self.timeline.selected)
        for f in self.frames:
            if f['name'] in names:
                self._set_decision(f, reject)
        self.store.save('decisions')
        self.timeline.clear_selection()
        self._decisions_changed()

    def _decisions_changed(self):
        row = self.table.currentIndex().row()
        cur = self._current_frame()
        self.model.dataChanged.emit(self.model.index(0, 0),
                                    self.model.index(len(self.model.frames) - 1, len(self.model.COLS) - 1))
        self.proxy.invalidateFilter()           # frames may leave the Kept / Rejected tab
        self._fill_groups()
        self._update_counts()
        self._update_timeline()
        if cur is None or self._current_frame() is not cur:   # it left the view: stay at the same position
            n = self.proxy.rowCount()
            if n:
                self.table.setCurrentIndex(self.proxy.index(min(max(row, 0), n - 1), 0))
        else:
            self.show_frame(cur)

    def _update_counts(self):
        if self.store:
            k = len(self.store.decisions)
            self.tabs.setTabText(len(VIEWS) - 1, f'My choices ({k})' if k else 'My choices')
        n = len(self.frames)
        nrej = sum(self.store.is_rejected(f) for f in self.frames) if self.store else 0
        nsus = sum(f['tier'] == 'suspect' and not self.store.is_rejected(f) for f in self.frames)
        self.lbl_counts.setText(f'  {n} frames · {nrej} reject · {nsus} suspect · {n - nrej - nsus} ok · '
                                f'showing {self.proxy.rowCount()} · {self._scan_time_text()}  ')

    def _scan_time_text(self):
        """Elapsed time while scanning; otherwise the last completed scan of this project."""
        if self.scan_thread and self.scan_thread.isRunning():
            return f'scanning {_duration(time.perf_counter() - self._scan_start)}'
        last = self.store.config.get('last_scan') if self.store else None
        if not last:   # measured by a version that did not record scan times, or never scanned
            return 'scan time not recorded' if self.frames else 'not scanned yet'
        verb = 'rescan' if last.get('rescan') else 'scan'
        return f'last {verb}: {last["frames"]} frames in {_duration(last["seconds"])} ({last["when"]})'

    # -- selection / preview ------------------------------------------------------------------
    def _current_frame(self):
        idx = self.table.currentIndex()
        if not idx.isValid():
            return None
        return self.model.frames[self.proxy.mapToSource(idx).row()]

    def _select_name(self, name):
        for r in range(self.proxy.rowCount()):
            src = self.proxy.mapToSource(self.proxy.index(r, 0)).row()
            if self.model.frames[src]['name'] == name:
                self.table.setCurrentIndex(self.proxy.index(r, 0))
                return

    def _row_changed(self, cur, _prev):
        f = self._current_frame()
        if not f:
            return
        if self.btn_compare.isChecked():        # moving to another frame ends compare mode
            self.btn_compare.blockSignals(True)
            self.btn_compare.setChecked(False)
            self.btn_compare.blockSignals(False)
        self.show_frame(f)
        self.timeline.set_current(f['name'])
        self._prefetch()

    def _load_preview(self, name):
        p = self._prev_cache.get(name)
        if p is None:
            p = self.store.load_preview(name)
            if p is not None:
                if len(self._prev_cache) > 150:      # ~6 MB each
                    self._prev_cache.pop(next(iter(self._prev_cache)))
                self._prev_cache[name] = p
        return p

    def _prefetch(self):
        """Load the next few previews in the background so blinking never waits on the disk."""
        r = self.table.currentIndex().row()
        names = []
        for k in range(1, 4):
            idx = self.proxy.index(r + k, 0)
            if idx.isValid():
                names.append(self.model.frames[self.proxy.mapToSource(idx).row()]['name'])
        names = [n for n in names if n not in self._prev_cache]
        if names:
            threading.Thread(target=lambda: [self._load_preview(n) for n in names], daemon=True).start()

    def show_frame(self, f, reference_for=None, placement=(False, 0.0, 0.0)):
        """Show frame f. reference_for=<frame>: f is shown as the comparison reference for that frame
        (blue border, and both use one shared stretch so the difference stays visible)."""
        p = self._load_preview(f['name'])
        if p is None:
            self.view.set_message('No preview cached for this frame — press Scan')
            return
        m = f['metrics']
        shape = m.get('shape') or [p.shape[0], p.shape[1]]
        if reference_for is not None:
            border = BORDER['reference']
        else:
            border = BORDER['reject'] if self.store.is_rejected(f) else \
                BORDER['suspect'] if f['tier'] == 'suspect' else None
        shared = self.btn_compare.isChecked() or reference_for is not None
        self.view.osc = self.store.osc
        flip, dx, dy = placement
        self.view.set_frame(f['name'], f['path'], p, self._lut(f, p, shared=shared), shape, border,
                            flip=flip, offset=(dx, dy))
        self._show_info(f, reference_for)

    # reason prefix -> (value to MINIMISE for the reference, description for the label)
    REF_KEYS = {
        'weight': (lambda x: -x['weight'], 'highest weight'),
        'FWHM': (lambda x: x['r_fwhm'], 'sharpest star cores (lowest FWHM)'),
        'HFR': (lambda x: x['r_hfr'], 'tightest stars (lowest HFR)'),
        'ecc': (lambda x: x['metrics'].get('ecc', np.nan), 'roundest stars (lowest eccentricity)'),
    }

    def _best_frame(self, f):
        """Reference for f: among clean frames of the same night + filter (OK and at least half the typical
        weight), the best one on the metric f was flagged for (its first reason; weight if none).
        Returns (frame, description) or (None, '')."""
        reason = (f['reasons'][0].split()[0] if f['reasons'] else 'weight')
        key, desc = self.REF_KEYS.get(reason, self.REF_KEYS['weight'])
        group = [x for x in self.frames if x['group'] == f['group'] and not self.store.is_rejected(x)]
        clean = [x for x in group if x['tier'] == 'ok' and x.get('weight', 0) >= 0.5] or group
        clean = [x for x in clean if np.isfinite(key(x))]
        return (min(clean, key=key), desc) if clean else (None, '')

    def set_compare(self, on):
        f = self._current_frame()
        if not f:
            return
        if not on:
            self.show_frame(f)
            return
        ref, self._ref_desc = self._best_frame(f)
        if ref is None or ref is f:
            self.btn_compare.blockSignals(True)
            self.btn_compare.setChecked(False)
            self.btn_compare.blockSignals(False)
            self.lbl_status.setText('This already is the best frame of its night + filter.' if ref is f
                                    else 'No reference frame available for this group.')
            return
        self._ref_note = ''
        placement = (False, 0.0, 0.0)
        pc, pr = self._load_preview(f['name']), self._load_preview(ref['name'])
        if pc is not None and pr is not None:
            key = (f['name'], ref['name'])
            if key not in self._align_cache:
                self._align_cache[key] = align(pc, pr)
            flip, dx, dy, score = self._align_cache[key]
            if score >= 0.3:
                b = (f['metrics'].get('shape') or pc.shape)[1] / pc.shape[1]    # preview -> sensor pixels
                placement = (flip, dx * b, dy * b)
                self._ref_note = ('rotated 180° (other side of the meridian) and aligned' if flip
                                  else 'aligned to your frame')
            else:
                self._ref_note = 'could not be aligned (too few matching stars)'
        self.show_frame(ref, reference_for=f, placement=placement)

    def _show_info(self, f, reference_for=None):
        m = f['metrics']
        if reference_for is not None:
            self.lbl_frame.setText(
                f'<b style="color:#4696ff">REFERENCE — {getattr(self, "_ref_desc", "best frame")} in '
                f'{f["group"]}</b> <i>{getattr(self, "_ref_note", "")}</i> (press R to flip back to '
                f'{reference_for["metrics"].get("date", "")[11:16]}, weight {_fmt_pct(reference_for["weight"])})'
                f'<br>{f["name"]} · weight {_fmt_pct(f["weight"])} · stars {m.get("nstars")} · '
                f'FWHM {m.get("fwhm", np.nan) * m.get("scale", np.nan):.1f}″ · HFR {_fmt_pct(f["r_hfr"])} · '
                f'ecc {m.get("ecc", np.nan):.2f}')
            return
        self.lbl_frame.setText(f'<b>{f["name"]}</b><br>{self.model.status(f).upper()}'
                               f': {why_text(f)} · '
                               f'stars {m.get("nstars")} · HFR {m.get("hfr", 0):.2f} · '
                               f'FWHM {m.get("fwhm", np.nan) * m.get("scale", np.nan):.1f}″ · '
                               f'ecc {m.get("ecc", np.nan):.2f} · SNR {m.get("snr", np.nan):.0f} · '
                               f'weight {_fmt_pct(f["weight"])} · sky {m.get("bg", 0):.0f}')

    def _lut(self, f, p=None, shared=False):
        """Autostretch off -> linear. Per frame -> this frame's own stretch. Default: one stretch per
        night+filter (from its median frame), so clouds and twilight stand out as brighter frames.
        shared=True (compare mode) always uses the per-night+filter stretch, so a frame and its
        reference are shown identically."""
        if not self.btn_stretch.isChecked():
            return LINEAR_LUT
        if not shared and not self.chk_linked.isChecked() and p is not None:
            sub = p[::4, ::4]
            med = float(np.median(sub))
            return stf_lut(med, float(np.median(np.abs(sub - med))))
        g = f['group']
        if g not in self._lut_cache:
            s = self.stats[g]
            ref = [x for x in self.frames if x['group'] == g and x['tier'] == 'ok'] or \
                  [x for x in self.frames if x['group'] == g]
            rp = self._load_preview(ref[len(ref) // 2]['name'])
            if rp is None:
                return stf_lut(s['bg'], max(s['bg'] * 0.02, 1))
            sub = rp[::4, ::4]
            med = float(np.median(sub))
            self._lut_cache[g] = stf_lut(med, float(np.median(np.abs(sub - med))))
        return self._lut_cache[g]

    # -- moving files ---------------------------------------------------------------------------
    def _sync_paths(self):
        self._paths.update({f['name']: f['path'] for f in self.frames})

    def move_rejects(self):
        n = movedialog.move_rejects(self, self.store, self.frames)
        self._sync_paths()
        if n:
            self.lbl_status.setText(f'Moved {n} file(s). Nothing was deleted — Restore… puts them back.')
        self.refresh()
        write_candidates(self.store, self.frames)

    def restore_rejects(self):
        n = movedialog.restore_all(self, self.store, self.frames)
        self._sync_paths()
        if n:
            self.lbl_status.setText(f'Restored {n} file(s) to their original folders.')
        self.refresh()
        write_candidates(self.store, self.frames)

    # -- blinking -------------------------------------------------------------------------------
    def step(self, d, wrap=False):
        n = self.proxy.rowCount()
        if not n:
            return
        r = self.table.currentIndex().row()
        r = (r + d) % n if wrap else min(max(r + d, 0), n - 1)
        self.table.setCurrentIndex(self.proxy.index(r, self.table.currentIndex().column() if r >= 0 else 0))
        self.table.scrollTo(self.proxy.index(r, 0))

    def set_playing(self, on):
        self.btn_play.setText('❚❚ Pause' if on else '▶ Play')
        if on:
            self.play_timer.start(int(1000 / self.fps.value()))
        else:
            self.play_timer.stop()

    def _stretch_changed(self, *_):
        self.chk_linked.setEnabled(self.btn_stretch.isChecked())
        f = self._current_frame()
        if f:
            self.show_frame(f)

    def _tab_changed(self, i):
        self.proxy.set_view(VIEWS[i][0])
        self.btn_clear_mine.setVisible(VIEWS[i][0] == 'mine')
        self._update_counts()

    def undo_current(self):
        """U: forget the manual choice for this frame (back to the automatic result)."""
        f = self._current_frame()
        if not f or not self.store or f['name'] not in self.store.decisions:
            return
        del self.store.decisions[f['name']]
        self.store.save('decisions')
        self._decisions_changed()

    def clear_my_choices(self):
        n = len(self.store.decisions) if self.store else 0
        if not n:
            return
        if QMessageBox.question(self, APP, f'Forget all {n} of your manual reject / keep choices in this project?\n\n'
                                'Every frame goes back to the automatic result. Files are not moved; '
                                'use Move rejects… / Restore… afterwards if needed.') != QMessageBox.Yes:
            return
        self.store.decisions.clear()
        self.store.save('decisions')
        self._decisions_changed()

    def toggle_current(self):
        """X: reject <-> keep. Going back to what stage 1 decided removes the manual override."""
        f = self._current_frame()
        if not f or not self.store:
            return
        self._set_decision(f, not self.store.is_rejected(f))
        self.store.save('decisions')
        self._decisions_changed()

    def closeEvent(self, e):
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.cancel()
            self.scan_thread.wait()
        if self.store:
            self.store.save()
        super().closeEvent(e)


def _dark_palette(app):
    app.setStyle(QStyleFactory.create('Fusion'))
    p = QPalette()
    base, alt, text = QColor(30, 30, 32), QColor(42, 42, 46), QColor(220, 220, 220)
    for role, col in ((QPalette.Window, alt), (QPalette.WindowText, text), (QPalette.Base, base),
                      (QPalette.AlternateBase, alt), (QPalette.Text, text), (QPalette.Button, alt),
                      (QPalette.ButtonText, text), (QPalette.ToolTipBase, alt), (QPalette.ToolTipText, text),
                      (QPalette.Highlight, QColor(60, 100, 170)), (QPalette.HighlightedText, QColor(255, 255, 255))):
        p.setColor(role, col)
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), 'icon.png')))
    if os.name == 'nt':   # own taskbar icon instead of python's
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('nightsift.app')
    _dark_palette(app)
    settings = QSettings(APP, APP)
    if not settings.value('last_project'):   # carry over from the pre-rename settings
        old = QSettings('astroblink', 'astroblink').value('last_project')
        if old:
            settings.setValue('last_project', old)
    w = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    sys.exit(app.exec())
