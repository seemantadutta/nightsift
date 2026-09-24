"""Per-frame quality metrics + preview generation. Runs in worker processes.

Speed strategy: detect stars on a 2x2-binned copy (4x fewer pixels), then measure
HFR / elongation on full-resolution cutouts of the brightest unsaturated stars.
"""
import os
import cv2
import numpy as np
import sep

from .fitsio import read_fits

METRICS_VERSION = 2   # bump when measurements change, so cached values get re-measured
PREVIEW_MAX_W = 2100   # preview width target (pixels)
STAMP_R = 12           # cutout radius for full-res star measurement
MAX_STARS = 500        # brightest stars used for HFR / shape

_oy, _ox = np.mgrid[-STAMP_R:STAMP_R + 1, -STAMP_R:STAMP_R + 1].astype(np.float32)


cv2.setNumThreads(1)  # one frame per process; parallelism comes from the process pool


def _bin(img, f, dtype=np.float32):
    """f x f average binning (OpenCV area resize == exact block mean for integer factors)."""
    h, w = img.shape
    out = cv2.resize(img[:h // f * f, :w // f * f], (w // f, h // f), interpolation=cv2.INTER_AREA)
    return out.astype(dtype, copy=False)


def make_preview(img):
    f = max(1, int(np.ceil(img.shape[1] / PREVIEW_MAX_W)))
    if img.dtype == np.uint16:
        return _bin(img, f, np.uint16) if f > 1 else img
    p = _bin(img, f) if f > 1 else img
    return np.clip(p, 0, 65535).astype(np.uint16)


def stamp_shapes(img, x, y):
    """Per-star measurements on full-res cutouts. Returns dict of arrays:
    hfr (flux-weighted mean radius, NINA's definition), elong (a/b from 2nd moments),
    fwhm (from the area above half maximum), flux (background-subtracted), noise (per-pixel,
    robust, from the cutout borders)."""
    R = STAMP_R
    h, w = img.shape
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    ok = (xi >= R) & (yi >= R) & (xi < w - R) & (yi < h - R)
    xi, yi = xi[ok], yi[ok]
    if len(xi) == 0:
        return None
    st = img[yi[:, None, None] + _oy.astype(int), xi[:, None, None] + _ox.astype(int)].astype(np.float32)
    border = np.concatenate([st[:, 0, :], st[:, -1, :], st[:, :, 0], st[:, :, -1]], axis=1)
    bmed = np.median(border, axis=1)
    noise = 1.4826 * np.median(np.abs(border - bmed[:, None]), axis=1)
    st = np.clip(st - bmed[:, None, None], 0, None)
    tot = st.sum(axis=(1, 2)) + 1e-9
    cx = (st * _ox).sum(axis=(1, 2)) / tot
    cy = (st * _oy).sum(axis=(1, 2)) / tot
    dx, dy = _ox - cx[:, None, None], _oy - cy[:, None, None]
    rr = np.sqrt(dx * dx + dy * dy)
    inr = rr <= R
    wi = st * inr
    swi = wi.sum(axis=(1, 2)) + 1e-9
    hfr = (wi * rr).sum(axis=(1, 2)) / swi
    mxx = (wi * dx * dx).sum(axis=(1, 2)) / swi
    myy = (wi * dy * dy).sum(axis=(1, 2)) / swi
    mxy = (wi * dx * dy).sum(axis=(1, 2)) / swi
    t1, t2 = (mxx + myy) / 2, np.sqrt(((mxx - myy) / 2) ** 2 + mxy ** 2)
    elong = np.sqrt((t1 + t2) / np.maximum(t1 - t2, 1e-6))
    peak = st.max(axis=(1, 2))
    fwhm = 2 * np.sqrt((st >= peak[:, None, None] / 2).sum(axis=(1, 2)) / np.pi)
    return dict(hfr=hfr, elong=elong, fwhm=fwhm, flux=swi, noise=noise)


def _trimmed_mean(a, lo=25, hi=75):
    """Mean of the middle 50%: robust like a median but not quantised (FWHM areas are integers)."""
    q1, q3 = np.percentile(a, [lo, hi])
    mid = a[(a >= q1) & (a <= q3)]
    return float(mid.mean()) if len(mid) else float(np.median(a))


def star_metrics(img, sat_level, det_bin=2):
    """det_bin: binning used for detection (2 for mono; 1 when img is already an OSC superpixel image)."""
    sep.set_extract_pixstack(3_000_000)
    sep.set_sub_object_limit(4096)
    b = _bin(img, det_bin) if det_bin > 1 else np.ascontiguousarray(img, dtype=np.float32)
    bkg = sep.Background(b, bw=64, bh=64)
    bmap = bkg.back()
    objs = sep.extract(b - bmap, 20, err=bkg.globalrms, minarea=4)
    out = dict(bg=float(bkg.globalback), noise=float(bkg.globalrms),
               bg_grad=float(np.percentile(bmap, 95) - np.percentile(bmap, 5)),
               nstars=0, hfr=np.nan, elong=np.nan, ecc=np.nan, fwhm=np.nan, flux=np.nan, snr=np.nan)
    if len(objs) == 0:
        return out
    peak = objs['peak'] + bmap[objs['y'].astype(int), objs['x'].astype(int)]
    ok = (objs['flag'] == 0) & (peak < sat_level) & (objs['b'] > 0.3)
    out['nstars'] = int(ok.sum())
    if ok.sum() == 0:
        return out
    idx = np.argsort(-np.where(ok, objs['flux'], 0))[:min(MAX_STARS, int(ok.sum()))]
    k, off = det_bin, (det_bin - 1) / 2
    st = stamp_shapes(img, objs['x'][idx] * k + off, objs['y'][idx] * k + off)
    if st is None:
        return out
    el = float(np.median(st['elong']))
    fwhm = _trimmed_mean(st['fwhm'])
    flux, noise = float(np.median(st['flux'])), float(np.median(st['noise']))
    # SNR a typical bright star reaches with optimal (PSF-weighted) extraction:
    #   SNR = F / (sigma_pix * sqrt(4 pi) * sigma_psf),  sigma_psf = FWHM / 2.355
    # Falls with clouds (F down), bright sky (sigma up) and blur (FWHM up).
    snr = flux / (max(noise, 1e-6) * np.sqrt(4 * np.pi) * max(fwhm, 0.5) / 2.3548)
    out.update(hfr=float(np.median(st['hfr'])), elong=el, ecc=float(np.sqrt(max(0.0, 1 - 1 / el ** 2))),
               fwhm=fwhm, flux=flux, noise_px=noise, snr=float(snr))
    return out


def _pixel_scale(hdr, osc):
    """Arcsec per (measured) pixel from XPIXSZ [um, includes binning] and FOCALLEN [mm]; nan if unknown."""
    try:
        s = 206.265 * float(hdr['XPIXSZ']) / float(hdr['FOCALLEN'])
        return s * 2 if osc else s
    except (KeyError, ValueError, ZeroDivisionError):
        return float('nan')


def analyze_file(path, osc=False):
    """Read one FITS, return (metrics dict, uint16 preview). One disk read per file.

    osc=True (colour camera): each 2x2 Bayer cell is averaged into one 'superpixel'
    luminance value first, so the colour pattern does not distort star shapes.
    HFR is then in superpixel units (only ratios within a group are used)."""
    hdr, img = read_fits(path)
    if osc:
        img = _bin(img, 2, img.dtype if img.dtype == np.uint16 else np.float32)
    st = os.stat(path)
    m = dict(v=METRICS_VERSION, file=os.path.abspath(path), size=st.st_size, mtime=st.st_mtime,
             imagetyp=hdr.get('IMAGETYP', ''), filter=hdr.get('FILTER', ''),
             exptime=float(hdr.get('EXPTIME', hdr.get('EXPOSURE', 0)) or 0),
             date=hdr.get('DATE-LOC', hdr.get('DATE-OBS', '')),
             alt=float(hdr.get('CENTALT', 'nan') or 'nan'),
             bayer=hdr.get('BAYERPAT', ''), osc=bool(osc), scale=_pixel_scale(hdr, osc))
    m['shape'] = list(img.shape)   # (h, w) of the measured image (superpixels for OSC)
    sat = 0.9 * 65535 if int(hdr['BITPIX']) == 16 else 0.9 * float(img.max())
    m.update(star_metrics(img, sat, det_bin=1 if osc else 2))
    return m, make_preview(img)
