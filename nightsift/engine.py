"""Shared scan / scoring / move logic used by both the CLI and the GUI."""
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .fitsio import read_header
from .metrics import analyze_file
from .scoring import evaluate, group_key

DEFAULT_WORKERS = 6
_CALIB_WORDS = ('DARK', 'FLAT', 'BIAS')


def _background_mode():
    """Worker initializer: low CPU + I/O + memory priority so scanning never makes the PC sluggish."""
    if os.name == 'nt':
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00100000)  # PROCESS_MODE_BACKGROUND_BEGIN
    else:
        os.nice(10)


def _worker(path, preview_path, osc):
    m, prev = analyze_file(path, osc)
    np.save(preview_path, prev)
    return m


def is_light(path):
    """Light frame per the IMAGETYP header; if that keyword is missing, anything inside a
    DARK/FLAT/BIAS-named folder is treated as calibration."""
    try:
        t = read_header(path)[0].get('IMAGETYP')
    except Exception:
        return False
    if t is None:
        dirs = os.path.dirname(os.path.abspath(path)).upper().replace('\\', '/').split('/')
        return not any(w in d for d in dirs for w in _CALIB_WORDS)
    t = t.upper()
    return 'LIGHT' in t or t in ('', 'OBJECT')


def looks_osc(path):
    """True if the FITS header declares a Bayer (colour) sensor."""
    try:
        return bool(read_header(path)[0].get('BAYERPAT'))
    except Exception:
        return False


def pending_lights(store, force=False, files=None, on_progress=None):
    """Light frames that still need measuring (new, changed, or measured in the other mono/OSC mode).
    force=True: every frame (full rescan: re-measure and rebuild all previews).
    on_progress(done, total) is called while file headers are being checked."""
    if force:
        store.skipped.clear()          # re-check calibration frames too
    if files is None:
        files = store.discover(include_rejected=True)
    todo = [f for f in files if force or (not store.is_current(f) and not store.is_skipped(f))]
    lights = []
    for i, f in enumerate(todo):
        if is_light(f):
            lights.append(f)
        else:
            store.skip(f)   # remember calibration frames so we never re-read their headers
        if on_progress and (i % 25 == 0 or i == len(todo) - 1):
            on_progress(i + 1, len(todo))
    return lights


def detect_osc(store, lights):
    """First scan of a project: set colour mode from the first light's header (BAYERPAT)."""
    if 'osc' not in store.config and lights:
        store.config['osc'] = looks_osc(lights[0])
        store.save('config')
    return store.osc


def scan_iter(store, workers=DEFAULT_WORKERS, cancelled=lambda: False, force=False, on_phase=None):
    """Measure pending frames (all frames if force). Yields (done, total, path, error_or_None) after
    each frame; metrics land in store.metrics as they arrive, replacing old values one by one, so a
    cancelled rescan leaves the not-yet-reached frames with their previous results.

    on_phase(phase, a, b) reports what happens before measuring starts:
      ('files', paths_list, None)  - discovery done (all FITS in project + reject folder)
      ('check', done, total)       - reading headers to pick out the light frames
      ('osc', is_osc, None)        - colour mode decided (first scan of a project)
      ('measure', 0, total)        - measuring starts"""
    phase = on_phase or (lambda *a: None)
    files = store.discover(include_rejected=True)
    phase('files', files, None)
    if cancelled():
        return
    todo = pending_lights(store, force, files, lambda d, t: phase('check', d, t))
    phase('osc', detect_osc(store, todo), None)
    if not todo or cancelled():
        store.save('metrics', 'skipped')
        return
    phase('measure', 0, len(todo))
    osc, done, last_save = store.osc, 0, time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_background_mode) as ex:
        futs = {ex.submit(_worker, f, store.preview_path(os.path.basename(f)), osc): f for f in todo}
        try:
            for fu in as_completed(futs):
                path, err = futs[fu], None
                try:
                    m = fu.result()
                    with store.lock:
                        store.metrics[os.path.basename(path)] = m
                except Exception as e:
                    err = repr(e)
                done += 1
                yield done, len(todo), path, err
                if time.perf_counter() - last_save > 30:
                    store.save('metrics', 'skipped')
                    last_save = time.perf_counter()
                if cancelled():
                    ex.shutdown(wait=True, cancel_futures=True)
                    break
        finally:
            store.save('metrics', 'skipped')


def build_frames(store, strictness, paths=None):
    """Frames present on disk (incl. the reject folder), with metrics + tier, in display order.
    paths: {basename: path} if already known (the GUI passes its list, so it never walks the disk)."""
    if paths is None:
        paths = {os.path.basename(p): p for p in store.discover(include_rejected=True)}
    with store.lock:
        ms = [m for n, m in store.metrics.items() if n in paths]
    result, stats = evaluate(ms, strictness)
    frames = []
    for m in ms:
        n = os.path.basename(m['file'])
        tier, reasons = result[n]
        frames.append(dict(name=n, path=paths[n], metrics=m, group=group_key(m), tier=tier, reasons=reasons))
    frames.sort(key=lambda f: (f['group'], f['metrics'].get('date', ''), f['name']))
    return frames, stats


def write_candidates(store, frames):
    """CSV of every flagged frame (what would be moved + suspects), for checking by hand."""
    path = os.path.join(store.dir, 'rejects.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['action', 'tier', 'reasons', 'time', 'group', 'stars', 'hfr', 'elong', 'sky', 'file'])
        for f in frames:
            rej, d = store.is_rejected(f), store.decisions.get(f['name'])
            if f['tier'] == 'ok' and not d:
                continue
            action = ('reject' if rej else 'keep') + (' (you)' if d else '')
            m = f['metrics']
            w.writerow([action, f['tier'], '; '.join(f['reasons']), m.get('date', '')[:19], f['group'],
                        m.get('nstars'), round(m.get('hfr') or 0, 2), round(m.get('elong') or 0, 3),
                        round(m.get('bg') or 0), f['path']])
    return path


def plan_moves(store, frames):
    """(to_move, to_restore): frames whose location does not match their reject/keep status."""
    to_move = [f for f in frames if store.is_rejected(f) and not store.in_reject_dir(f['path'])]
    to_restore = [f for f in frames if not store.is_rejected(f) and store.in_reject_dir(f['path'])]
    return to_move, to_restore


def execute_moves(store, to_move, to_restore):
    """Move files (never delete). Updates each frame's 'path'. Returns list of (name, error)."""
    errors = []
    for f, fn in [(f, store.move_to_rejected) for f in to_move] + [(f, store.restore) for f in to_restore]:
        try:
            f['path'] = fn(f['path'])
        except OSError as e:
            errors.append((f['name'], repr(e)))
    return errors
