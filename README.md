# NightSift

Fast culling of astrophotography subframes (FITS lights).

NightSift works in two stages:

1. **Auto-flag.** It measures every light frame in the background and flags the bad ones. Bad means low weight (clouds, haze, twilight), defocus or "donut" stars, bloated stars, or trailed stars (guiding failure, wind, cable snag).
2. **Blink review.** A fast viewer lets you confirm or overrule the flags. Then it moves the rejects out of the project so that a single-folder WBPP run only sees good frames.

NightSift **never deletes anything**. It only moves files, and every move can be undone with Restore.

Built for a mono camera (LRGB + narrowband) with NINA capture and PixInsight WBPP. It also handles OSC (colour) data.

## Install

Requires Python 3.10+.

```
pip install -e .
```

Dependencies: numpy, sep, opencv-python, PySide6.

## Run

**Desktop app:**

```
python -m nightsift.app [project folder]
```

On Windows, use `pythonw.exe -m nightsift.app` for a shortcut without a console window.

**Command line:**

```
nightsift scan    PROJECT   # stage 1: measure and flag (moves nothing)
nightsift blink   PROJECT   # stage 2: blink review, then move rejects after asking
nightsift         PROJECT   # scan + blink
nightsift watch   PROJECT   # keep scanning new frames during capture
nightsift restore PROJECT   # move every rejected frame back
```

Useful flags:

| Flag | Effect |
|---|---|
| `-s/--strictness 1.2` | Flag more (lower values flag less) |
| `-j 6` | Number of scan workers |
| `-r/--reject-dir DIR` | Where rejected frames go |
| `--osc` / `--no-osc` | Force colour or mono handling |
| `--rescan` | Measure every frame again |
| `-n/--dry-run` | Show what would be moved, touch nothing |
| `--flagged` / `--suspect` | Blink only those frames |

## How frames are judged

Frames are compared only with frames of the **same night + filter + exposure**. A frame is flagged if **any** of these checks fails:

| Check | Reject | Suspect | Catches |
|---|---|---|---|
| Weight = (SNR / group median)² | < 10 % | < 40 % | Clouds, haze, bright sky, blur |
| FWHM / group median | > 2.0× | > 1.5× | Defocus, donut stars |
| HFR / group median | > 1.8× | > 1.3× | Bloated halos, soft seeing |
| Eccentricity (must also exceed the group's typical value by 0.15 / 0.08) | > 0.70 | > 0.55 | Trailing |

The strictness slider scales these limits. The defaults were tuned so that every frame rejected by hand in two real datasets is flagged at strictness 1.0.

## Files it creates

Each project gets a `.nightsift/` cache folder with these contents:

- metrics
- your decisions
- move history
- previews
- `rejects.csv`
- `moves.log`

Rejected frames go outside the project, to `<project>_rejected/<night folder>/`, so that WBPP does not pick them up. You can change the location in the app or with `-r`.

## Desktop app shortcuts

| Key | Action |
|---|---|
| ← / → | Previous / next frame |
| Space | Play / pause |
| X / Delete | Reject or keep |
| U | Undo my choice |
| R | Compare to the best frame of the group, aligned across meridian flips |
| A | Autostretch |
| F / 1 | Fit / 1:1 zoom |
| + / − | Zoom |
