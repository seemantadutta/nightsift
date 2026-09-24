"""Stage 1 rules: flag frames that are outliers relative to their own group.

A group is one night + filter + exposure, so each frame is only compared with
frames taken under comparable conditions.

Two tiers:
  reject  - clear-cut: auto-rejected
  suspect - borderline: shown for a human decision, kept by default

Checks (each relative to the group's median frame):
  weight = (SNR / median SNR)^2 - roughly the frame's share of signal in a stack. Falls with
           clouds/haze (dimmer stars), bright sky (more noise) and blur. Replaces separate
           star-count and sky rules, which it subsumes.
  FWHM   - core size of stars. Catches defocus ("donut" stars), which HFR largely misses.
  HFR    - half-flux radius. Catches bloated halos (seeing, soft focus) with a normal core.
  ecc    - eccentricity: above an absolute limit AND clearly above the group's typical value.
           Trailing: guiding failure, wind, cable snag.

Tuned on 949 manually culled frames (6 nights LRGB): every manual reject is flagged; the
extra auto-rejects were verified by eye (donut stars, twilight, very low weight).
"""
import os
from datetime import datetime, timedelta
import numpy as np

#                      reject  suspect
WEIGHT_BELOW = (0.10, 0.40)   # (SNR / median)^2
FWHM_RATIO_ABOVE = (2.00, 1.50)
HFR_RATIO_ABOVE = (1.80, 1.30)
ECC_ABOVE = (0.70, 0.55)      # PixInsight scale; 0.70 ~ a/b 1.4, 0.55 ~ a/b 1.2
ECC_MARGIN = (0.15, 0.08)     # ...and must also exceed the group's typical ecc by this much, so a rig's
                              # normal slight ovalness is never flagged, however high the strictness


def _nanmedian(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float('nan')


def night_of(m):
    try:
        t = datetime.fromisoformat(m.get('date', '')[:19])
        return (t - timedelta(hours=12)).strftime('%Y-%m-%d')
    except ValueError:
        return 'unknown'


def group_key(m):
    return f'{night_of(m)} {m.get("filter") or "?"} {m.get("exptime", 0):g}s'


def _ecc_scaled(e, s):
    """Strictness for eccentricity: scale the axis ratio's excess over 1, convert back."""
    el = 1 / np.sqrt(max(1 - e * e, 1e-9))
    el = 1 + (el - 1) / s
    return float(np.sqrt(1 - 1 / el ** 2))


def _limits(strictness):
    """strictness > 1 tightens every threshold toward 'perfect', < 1 loosens it."""
    s = max(strictness, 1e-3)
    r = lambda v: v ** (1 / s)   # pulls ratios toward 1.0 as strictness rises
    return [dict(weight=r(WEIGHT_BELOW[t]), fwhm=r(FWHM_RATIO_ABOVE[t]), hfr=r(HFR_RATIO_ABOVE[t]),
                 ecc=_ecc_scaled(ECC_ABOVE[t], s)) for t in (0, 1)]


def _cause(rs, rb):
    """Why the weight is low, in words the user can check."""
    c = []
    if rb > 1.5:
        c.append(f'sky {rb:.1f}x')
    if rs < 0.7:
        c.append(f'stars {rs:.0%}')
    return f' ({", ".join(c)})' if c else ''


def classify(m, med, limits):
    """Return (tier, reasons) for one frame given its group medians."""
    if not m.get('nstars') or not np.isfinite(m.get('hfr', np.nan)):
        return 'reject', ['no stars']
    rs = m['nstars'] / med['nstars'] if med['nstars'] else 1.0
    rb = m['bg'] / med['bg'] if med['bg'] > 0 else 1.0
    w = (m['snr'] / med['snr']) ** 2 if np.isfinite(m.get('snr', np.nan)) and med.get('snr') else np.nan
    rf = m['fwhm'] / med['fwhm'] if np.isfinite(m.get('fwhm', np.nan)) and med.get('fwhm') else np.nan
    rh = m['hfr'] / med['hfr']
    ecc = m.get('ecc', np.nan)
    med_ecc = med.get('ecc', np.nan)
    for t, (tier, lim) in enumerate(zip(('reject', 'suspect'), limits)):
        ecc_lim = max(lim['ecc'], med_ecc + ECC_MARGIN[t]) if np.isfinite(med_ecc) else lim['ecc']
        r = []
        if np.isfinite(w) and w < lim['weight']:
            r.append(f'weight {w:.0%}{_cause(rs, rb)}')
        if np.isfinite(rf) and rf > lim['fwhm']:
            r.append(f'FWHM {rf:.0%} (defocus?)' if rf > 2 * rh / 1.3 else f'FWHM {rf:.0%}')
        if rh > lim['hfr']:
            r.append(f'HFR {rh:.0%}')
        if np.isfinite(ecc) and ecc > ecc_lim:
            r.append(f'ecc {ecc:.2f}')
        if r:
            return tier, r
    return 'ok', []


def evaluate(ms, strictness=1.0):
    """Return ({name: (tier, [reasons])}, {group: median stats})."""
    limits = _limits(strictness)
    groups = {}
    for m in ms:
        groups.setdefault(group_key(m), []).append(m)
    result, stats = {}, {}
    for g, gm in groups.items():
        med = {k: _nanmedian([m.get(k, np.nan) for m in gm])
               for k in ('nstars', 'hfr', 'elong', 'ecc', 'bg', 'snr', 'fwhm')}
        stats[g] = med
        for m in gm:
            result[os.path.basename(m['file'])] = classify(m, med, limits)
    return result, stats
