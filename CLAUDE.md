# NightSift — notes for Claude

A desktop and CLI tool that culls astrophotography FITS subs. Stage 1 auto-flags bad frames and stage 2 is a blink review. The user is an amateur astrophotographer with this setup:

- ZWO ASI294MM Pro mono, LRGB + narrowband
- NINA capture
- PixInsight WBPP, run on ONE project folder

See README.md for user-facing behaviour.

## Layout (`nightsift/`)

| File | Role |
|---|---|
| `fitsio.py` | Minimal fast FITS reader (uint16 fast path: in-place byteswap + XOR 0x8000; one disk read per file) |
| `metrics.py` | Per-frame measurement (`analyze_file`). Detects stars with sep on a 2×2-binned image, then measures HFR/FWHM/ecc/flux on full-res stamps and derives SNR. Also writes the binned uint16 preview. Bump `METRICS_VERSION` when the measurements change, so caches get invalidated. |
| `scoring.py` | Group rules and thresholds. Group = night (date − 12 h) + filter + exposure. Tiers are reject / suspect / ok, and a frame is flagged if ANY check fails. `strictness` scales the ratios as `v**(1/s)`. |
| `store.py` | Per-project cache in `<project>/.nightsift/`: metrics.json, decisions.json, moved.json, config.json, skipped.json, previews/*.npy, rejects.csv, moves.log. Also handles moving files to and from the reject folder. |
| `engine.py` | Scan pipeline (`scan_iter`, phases files/check/osc/measure) using a ProcessPoolExecutor at background priority, plus `build_frames`, `plan_moves` and `execute_moves`. |
| `cli.py` | `nightsift scan/blink/watch/restore/apply/all`. |
| `app.py` | GUI launcher (`python -m nightsift.app`). It must NOT import Qt at top level, because the scan worker processes re-import it. |
| `gui.py` | Main PySide6 window: group table, tabbed frame table, viewer, timeline, compare-to-best (R). |
| `imageview.py` | QGraphicsView viewer: zoom/pan, STF LUT, full-res load on zoom, flip/offset display for compare mode. |
| `timeline.py` | Metric-vs-time plot: Hubble palette colours, shaded bad zones, drag-select. |
| `align.py` | Aligns previews for compare mode: detects a meridian flip (180°) and the dither shift with cv2.matchTemplate. |
| `movedialog.py` | Preview dialogs for Move rejects… and Restore…. |
| `viewer.py` | Legacy OpenCV CLI blinker, plus `stf_lut`, which the GUI uses. |

## Hard rules

- **Never delete user files.** Only move them, and always keep a way to restore them.
- **Rejects go OUTSIDE the project**, to `<project>_rejected/<night folder>/`, because WBPP ingests the whole project folder.
- **Skip calibration frames.** Only LIGHT frames are scanned; darks, flats and bias are ignored.
- **No disk I/O on the GUI thread.** Project folders can be on a spinning HDD (L:), and a folder walk on the UI thread freezes the app. Use the QThreads (OpenThread, ScanThread), and use targeted `store.save('config'|'decisions'|…)` calls.
- **Mind memory use.** C: is nearly full, so commit memory is low. Keep the uint16 path, sep pixstack 3M and the default of 6 workers.
- **Judge frames by measured data, not by how the stretch looks.**

## User preferences

- Prefers checkboxes over toggle buttons.
- Don't browse the user's drives (e.g. `L:`) unprompted.

## Validation data (not in git)

- `testdata/`: VDB152 subset.
- `testdata3/`: IC 1590 OIII. The user's rejects are in `testdata3/badpacman`, and all of them are flagged at strictness 1.0.
- Full VDB152 labels: 949 frames with 27 manual rejects, all flagged at 1.0.

Re-check both datasets after changing thresholds or metrics.

## Dev notes

- Windows. Python at `C:\Python314`.
- The desktop shortcut runs `pythonw.exe -m nightsift.app` with this repo as the working directory.
- Tests override QSettings('NightSift','NightSift') `last_project`. Reset it to `L:\PI_M3\LDN1251` afterwards.
- Write files as UTF-8 explicitly (`encoding='utf-8'`), because the default codec is cp1252.
- There is no test suite yet. Verify changes with scripts against the test data.
