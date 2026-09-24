"""On-disk state for a scanned project folder, kept in <root>/.nightsift/.

Everything is keyed by file *basename* (capture software puts a timestamp/sequence
in it), so entries survive files being moved to the reject folder and back.

Rejected frames go OUTSIDE the project, so the project folder can be handed to
WBPP as-is. Layout:  <reject_dir>/<top-level folder under root>/<file>
e.g. L:/PI_M3/VDB152/NIGHT_1/LIGHT/L/x.fits -> L:/PI_M3/VDB152_rejected/NIGHT_1/x.fits
"""
import json
import os
import shutil
import threading
import numpy as np

from .metrics import METRICS_VERSION

CACHE_DIR = '.nightsift'
LEGACY_CACHE_DIR = '.astroblink'   # name used before the rename; migrated on open
FITS_EXT = ('.fit', '.fits', '.fts')
_FRAME_TYPE_DIRS = {'LIGHT', 'LIGHTS', 'LIGHT_FRAMES'}


def _atomic_write_json(path, obj, compact=False):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=None if compact else 1, separators=(',', ':') if compact else None)
    os.replace(tmp, path)


_PARTS = ('metrics', 'decisions', 'moved', 'config', 'skipped')


def _under(path, folder):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(folder)]) == os.path.abspath(folder)
    except ValueError:  # different drives
        return False


class Store:
    def __init__(self, root, reject_dir=None):
        self.root = os.path.abspath(root)
        self.lock = threading.RLock()   # scan thread writes metrics while the GUI reads them
        self.dir = os.path.join(self.root, CACHE_DIR)
        legacy = os.path.join(self.root, LEGACY_CACHE_DIR)
        if not os.path.exists(self.dir) and os.path.isdir(legacy):
            os.rename(legacy, self.dir)     # keep measurements + decisions from before the rename
        self.prev_dir = os.path.join(self.dir, 'previews')
        os.makedirs(self.prev_dir, exist_ok=True)
        self.metrics = self._load('metrics.json')      # name -> metrics dict
        self.decisions = self._load('decisions.json')  # name -> 'reject' | 'keep' (manual overrides)
        self.moved = self._load('moved.json')          # name -> original path of files now in reject_dir
        self.skipped = self._load('skipped.json')      # name -> size of non-light (calibration) files
        self.config = self._load('config.json')
        if reject_dir:
            self.config['reject_dir'] = os.path.abspath(reject_dir)
        default_reject = os.path.join(os.path.dirname(self.root), os.path.basename(self.root) + '_rejected')
        stored = self.config.get('reject_dir')
        if stored and not self.moved and not os.path.isdir(os.path.dirname(stored)):
            stored = None   # project was moved/renamed and nothing is parked in the old reject folder
        self.reject_dir = stored or default_reject
        self.config['reject_dir'] = self.reject_dir

    def _load(self, name):
        p = os.path.join(self.dir, name)
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
        return {}

    def save(self, *parts):
        """Write the given parts ('metrics', 'decisions', 'moved', 'config', 'skipped'; default all).
        Each dict is snapshotted under the lock and written outside it, so a slow disk never blocks
        the other thread for long. The GUI saves only what changed (e.g. 'decisions' for X)."""
        for name in parts or _PARTS:
            with self.lock:
                snap = dict(getattr(self, name))
            _atomic_write_json(os.path.join(self.dir, name + '.json'), snap, compact=name == 'metrics')

    def is_rejected(self, f):
        """Manual decision wins; otherwise only the auto 'reject' tier is rejected."""
        d = self.decisions.get(f['name'])
        return d == 'reject' if d else f['tier'] == 'reject'

    def preview_path(self, name):
        return os.path.join(self.prev_dir, name + '.npy')

    def load_preview(self, name):
        p = self.preview_path(name)
        return np.load(p) if os.path.exists(p) else None

    # -- file discovery ---------------------------------------------------
    def discover(self, include_rejected=False):
        """FITS files in the project (plus, optionally, ones this project moved to reject_dir)."""
        out = []
        for dp, dns, fns in os.walk(self.root):
            dns[:] = [d for d in dns if not d.startswith('.') and not _under(os.path.join(dp, d), self.reject_dir)]
            out += [os.path.join(dp, f) for f in fns if f.lower().endswith(FITS_EXT)]
        if include_rejected:
            out += [p for p in (self.rejected_path(n) for n in self.moved) if p and os.path.exists(p)]
        return sorted(out)

    @property
    def osc(self):
        return bool(self.config.get('osc', False))

    @property
    def strictness(self):
        return float(self.config.get('strictness', 1.0))

    def is_current(self, path):
        """Cached metrics exist for this exact file, measured in the current mono/OSC mode."""
        m = self.metrics.get(os.path.basename(path))
        if not m or bool(m.get('osc', False)) != self.osc or m.get('v') != METRICS_VERSION:
            return False
        st = os.stat(path)
        return m.get('size') == st.st_size and abs(m.get('mtime', 0) - st.st_mtime) < 1

    def is_skipped(self, path):
        return self.skipped.get(os.path.basename(path)) == os.path.getsize(path)

    def skip(self, path):
        self.skipped[os.path.basename(path)] = os.path.getsize(path)

    # -- moving files -----------------------------------------------------
    def in_reject_dir(self, path):
        return _under(path, self.reject_dir)

    def bucket(self, original):
        """Sub-folder of reject_dir for a file: the top-level folder under the project root
        (NIGHT_1, DATA_1, ...). Files directly in root, or when root is itself a single night
        (root/LIGHT/...), use the root folder's own name."""
        rel = os.path.relpath(original, self.root).split(os.sep)
        if len(rel) < 2 or rel[0].upper() in _FRAME_TYPE_DIRS:
            return os.path.basename(self.root)
        return rel[0]

    def rejected_path(self, name):
        orig = self.moved.get(name)
        return os.path.join(self.reject_dir, self.bucket(orig), name) if orig else None

    def planned_reject_path(self, path):
        return os.path.join(self.reject_dir, self.bucket(path), os.path.basename(path))

    def move_to_rejected(self, path):
        name = os.path.basename(path)
        dst = self.planned_reject_path(path)
        if os.path.exists(dst):
            raise FileExistsError(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(path, dst)
        self.moved[name] = path
        self._log_move(path, dst)
        self.save('moved')
        return dst

    def restore(self, path):
        """Move a rejected file back to where it came from."""
        name = os.path.basename(path)
        dst = self.moved.get(name)
        if not dst:
            raise KeyError(f'no original location recorded for {name}')
        if os.path.exists(dst):
            raise FileExistsError(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(path, dst)
        del self.moved[name]
        self._log_move(path, dst)
        self.save('moved')
        return dst

    def _log_move(self, src, dst):
        with open(os.path.join(self.dir, 'moves.log'), 'a') as fh:
            fh.write(f'{src}\t{dst}\n')
