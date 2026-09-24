"""Preview-and-confirm dialogs for moving rejects out of the project and restoring them."""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox, QHeaderView, QLabel,
                               QMessageBox, QProgressDialog, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .engine import execute_moves, plan_moves


class _PlanDialog(QDialog):
    def __init__(self, parent, title, intro, rows, ok_text, extra=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 560)
        v = QVBoxLayout(self)
        lbl = QLabel(intro)
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        t = QTableWidget(len(rows), 4)
        t.setHorizontalHeaderLabels(['Action', 'File', 'Why', 'Goes to'])
        t.verticalHeader().hide()
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setWordWrap(False)
        t.setTextElideMode(Qt.ElideMiddle)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                it = QTableWidgetItem(val)
                if c == 3:
                    it.setToolTip(val)
                t.setItem(r, c, it)
        h = t.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Interactive)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        t.setColumnWidth(1, 380)
        v.addWidget(t, 1)
        if extra:
            v.addWidget(extra)
        bb = QDialogButtonBox()
        bb.addButton(ok_text, QDialogButtonBox.AcceptRole)
        bb.addButton(QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


def _run(parent, store, to_move, to_restore):
    """Execute with a progress dialog (moves across drives copy the file first)."""
    n = len(to_move) + len(to_restore)
    prog = QProgressDialog('Moving files…', None, 0, n, parent)
    prog.setWindowModality(Qt.WindowModal)
    prog.setMinimumDuration(300)
    errors, done = [], 0
    for batch, restore in ((to_move, False), (to_restore, True)):
        for f in batch:
            errors += execute_moves(store, [] if restore else [f], [f] if restore else [])
            done += 1
            prog.setValue(done)
            QApplication.processEvents()
    prog.close()
    if errors:
        QMessageBox.warning(parent, 'NightSift', f'{len(errors)} file(s) could not be moved:\n\n' +
                            '\n'.join(f'{n}: {e}' for n, e in errors[:15]))
    return n - len(errors)


def move_rejects(parent, store, frames):
    """Preview what 'Move rejects' will do; move on confirm. Returns number of files moved (0 if cancelled)."""
    to_move, to_restore = plan_moves(store, frames)
    if not to_move and not to_restore:
        QMessageBox.information(parent, 'NightSift', 'Nothing to move: every rejected frame is already in\n'
                                f'{store.reject_dir}')
        return 0
    rows = [('move out', f['name'], ', '.join(f['reasons']) or 'your choice (X)',
             os.path.dirname(store.planned_reject_path(f['path']))) for f in to_move]
    rows += [('bring back', f['name'], 'you changed it to keep', os.path.dirname(store.moved.get(f['name'], '')))
             for f in to_restore]
    intro = (f'<b>{len(to_move)}</b> rejected frame(s) will be moved out of the project into '
             f'<b>{store.reject_dir}</b> (one folder per night).'
             + (f'<br><b>{len(to_restore)}</b> frame(s) you now want to keep will be moved back.' if to_restore else '')
             + '<br>Nothing is deleted — <i>Restore…</i> puts everything back.')
    d = _PlanDialog(parent, 'Move rejects', intro, rows, f'Move {len(rows)} file(s)')
    if d.exec() != QDialog.Accepted:
        return 0
    return _run(parent, store, to_move, to_restore)


def restore_all(parent, store, frames):
    """Preview + restore every frame currently in the reject folder."""
    inside = [f for f in frames if store.in_reject_dir(f['path'])]
    if not inside:
        QMessageBox.information(parent, 'NightSift', 'Nothing to restore: the reject folder has no frames '
                                'from this project.')
        return 0
    rows = [('bring back', f['name'], ', '.join(f['reasons']) or 'your choice (X)',
             os.path.dirname(store.moved.get(f['name'], ''))) for f in inside]
    forget = QCheckBox('Also forget my manual reject / keep choices (start fresh from the automatic results)')
    intro = (f'<b>{len(inside)}</b> frame(s) will be moved back from <b>{store.reject_dir}</b> '
             'to their original folders.<br>They stay marked as rejects, so they show up red again; '
             'press X on any you want to keep.')
    d = _PlanDialog(parent, 'Restore rejected frames', intro, rows, f'Restore {len(inside)} file(s)', forget)
    if d.exec() != QDialog.Accepted:
        return 0
    n = _run(parent, store, [], inside)
    if forget.isChecked():
        store.decisions.clear()
        store.save('decisions')
    return n
