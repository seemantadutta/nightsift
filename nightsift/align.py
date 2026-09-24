"""Align a reference frame to the current one for blinking: detects a meridian flip (180°
rotation) and the shift between exposures (dither, re-centring), from the previews alone."""
import cv2
import numpy as np


def _prep(p, f):
    """Downsample by f, flatten the background, clip bright stars, normalise."""
    a = p.astype(np.float32)
    if f > 1:
        a = cv2.resize(a, (a.shape[1] // f, a.shape[0] // f), interpolation=cv2.INTER_AREA)
    a -= cv2.GaussianBlur(a, (0, 0), 12)
    a = np.clip(a, 0, np.percentile(a, 99.7))
    return (a - a.mean()) / (a.std() + 1e-6)


def _match(img, ref, margin):
    """Best placement of ref's centre crop inside img. Returns (score, dx, dy) in img pixels:
    ref pixel (u, v) lands on img pixel (u + dx, v + dy)."""
    t = np.ascontiguousarray(ref[margin:-margin, margin:-margin])
    r = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(r)
    return score, x - margin, y - margin


def align(cur_preview, ref_preview, coarse=4, margin_frac=0.12):
    """Return (flip, dx, dy, score): show the reference rotated 180° if flip, then shifted by (dx, dy)
    preview pixels, to line up with the current frame. score < ~0.3 means no reliable match."""
    h, w = cur_preview.shape
    m = int(min(h, w) * margin_frac) // coarse
    a = _prep(cur_preview, coarse)
    best = None
    for flip in (False, True):
        b = _prep(ref_preview[::-1, ::-1] if flip else ref_preview, coarse)
        s, dx, dy = _match(a, b, m)
        if best is None or s > best[0]:
            best = (s, flip, dx, dy)
    s, flip, dx, dy = best
    # refine at full preview resolution within a few pixels of the coarse answer
    A = _prep(cur_preview, 1)
    B = _prep(ref_preview[::-1, ::-1] if flip else ref_preview, 1)
    M = m * coarse
    X, Y = dx * coarse, dy * coarse
    pad = coarse + 2
    y0, x0 = max(M + Y - pad, 0), max(M + X - pad, 0)
    t = np.ascontiguousarray(B[M:-M, M:-M])
    win = A[y0:y0 + t.shape[0] + 2 * pad, x0:x0 + t.shape[1] + 2 * pad]
    if win.shape[0] >= t.shape[0] and win.shape[1] >= t.shape[1]:
        r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, s2, _, (x, y) = cv2.minMaxLoc(r)
        X, Y, s = x0 + x - M, y0 + y - M, s2
    return flip, X, Y, float(s)
