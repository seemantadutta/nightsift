# NightSift — notes for Claude

A desktop and CLI tool that culls astrophotography FITS subs. Stage 1 auto-flags bad frames and stage 2 is a blink review. It targets a NINA capture → PixInsight WBPP workflow, where WBPP is run on ONE project folder.

See README.md for user-facing behaviour.

## Layout (`nightsift/`)

| File | Role |
|---|---|
| `fitsio.py` | Minimal fast FITS reader (uint16 fast path: in-place byteswap + XOR 0x8000; one disk read per file) |
| `metrics.py` | Per-frame measurement (`analyze_file`). Detects stars with sep on a 2×2-binned image, then measures HFR/FWHM/ecc/flux on full-res stamps and derives SNR. Also writes the binned uint16 preview. Bump `METRICS_VERSION` when the measurements change, so caches get invalidated. |
| `scoring.py` | Group rules and thresholds. Group = night (date − 12 h) + filter + exposure. Tiers are reject / suspect / ok, and a frame is flagged if ANY check fails. `strictness` scales the ratios as `v**(1/s)`. |
| `store.py` | Per-project cache in `<project>/.nightsift/`: metrics.json, decisions.json, moved.json, config.json, skipped.json, previews/*.npy, rejects.csv, moves.log. Also handles moving files to and from the reject folder. |
| `engine.py` | Scan pipeline (`scan_iter`, phases files/check/osc/measure) using a ProcessPoolExecutor at background priority, plus `build_frames`, `plan_moves`, `execute_moves` and the integration-time summary. |
| `cli.py` | `nightsift scan/blink/watch/restore/apply/all`. |
| `app.py` | GUI launcher (`python -m nightsift.app`). It must NOT import Qt at top level, because the scan worker processes re-import it. |
| `gui.py` | Main PySide6 window: group table, tabbed frame table, viewer, timeline, compare-to-best (R), watch mode. |
| `imageview.py` | QGraphicsView viewer: zoom/pan, STF LUT, full-res load on zoom, flip/offset display for compare mode. |
| `timeline.py` | Metric-vs-time plot: Hubble palette colours, shaded bad zones, drag-select. |
| `align.py` | Aligns previews for compare mode: detects a meridian flip (180°) and the dither shift with cv2.matchTemplate. |
| `movedialog.py` | Preview dialogs for Move rejects… and Restore…. |
| `viewer.py` | Legacy OpenCV CLI blinker, plus `stf_lut`, which the GUI uses. |

## Hard rules

- **Never delete user files.** Only move them, and always keep a way to restore them.
- **Rejects go OUTSIDE the project**, to `<project>_rejected/<night folder>/`, because WBPP ingests the whole project folder.
- **Skip calibration frames.** Only LIGHT frames are scanned; darks, flats and bias are ignored.
- **No disk I/O on the GUI thread.** Project folders are often on spinning HDDs, and a folder walk on the UI thread freezes the app. Use the QThreads (OpenThread, ScanThread), and use targeted `store.save('config'|'decisions'|…)` calls.
- **Mind memory use.** Keep the uint16 path, sep pixstack 3M and the default of 6 workers.
- **Judge frames by measured data, not by how the stretch looks.**
- UI: use checkboxes rather than toggle buttons.

## Validation data (not in git)

- `testdata/`, `testdata3/`: labelled datasets with manual rejects. At strictness 1.0 every manual reject must be flagged.

Re-check them after changing thresholds or metrics.

## Dev notes

- Windows first. Requires Python 3.10+.
- GUI tests overwrite QSettings('NightSift','NightSift') `last_project`, `recent_projects` and `watch_seconds`. Save them before a test and restore them afterwards.
- Write files as UTF-8 explicitly (`encoding='utf-8'`), because the default codec on Windows is cp1252.
- There is no test suite yet. Verify changes with scripts against the test data.
- Changes go through pull requests; don't push to `main` directly.
