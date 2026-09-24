"""nightsift command line.

  nightsift scan  PATH   stage 1: measure every light, auto-flag bad frames (moves nothing)
  nightsift blink PATH   stage 2: blink review, then (after asking) move rejects out
  nightsift PATH         scan + blink
  nightsift watch PATH   keep scanning new frames as they arrive (run during capture)
  nightsift restore PATH move every rejected frame back into the project

Rejected frames are moved OUTSIDE the project (default: <project>_rejected next to it,
change with --reject-dir; remembered per project), one sub-folder per night folder.
"""
import argparse
import os
import sys
import time

from .engine import (DEFAULT_WORKERS, build_frames, execute_moves, plan_moves, scan_iter,
                     write_candidates)
from .store import Store


def scan(store, workers, quiet=False, force=False):
    t0, n = time.perf_counter(), 0
    for done, total, path, err in scan_iter(store, workers, force=force):
        n = total
        if err:
            print(f'\n  ! {os.path.basename(path)}: {err}')
        if not quiet:
            el = time.perf_counter() - t0
            print(f'\r  scanning {done}/{total}  {done / el:.1f} frames/s  ETA {el / done * (total - done):.0f}s   ',
                  end='', flush=True)
    if n and not quiet:
        print()
    return n


def report(frames, stats, store):
    groups = {}
    for f in frames:
        groups.setdefault(f['group'], []).append(f)
    print()
    print(f'  {"group":<26}{"frames":>7}{"reject":>8}{"suspect":>9}{"stars":>8}{"HFR":>7}{"elong":>7}{"sky":>8}')
    for g, fs in groups.items():
        s = stats[g]
        nr = sum(f['tier'] == 'reject' for f in fs)
        ns = sum(f['tier'] == 'suspect' for f in fs)
        print(f'  {g:<26}{len(fs):>7}{nr:>8}{ns:>9}{s["nstars"]:>8.0f}{s["hfr"]:>7.2f}{s["elong"]:>7.2f}{s["bg"]:>8.0f}')
    for tier, mark in (('reject', 'x'), ('suspect', '?')):
        fl = [f for f in frames if f['tier'] == tier]
        print(f'\n  {len(fl)} {tier}' + (' (auto)' if tier == 'reject' else ' (review in blink)'))
        for f in fl:
            d = store.decisions.get(f['name'])
            tag = f'  [you: {d}]' if d else ''
            print(f'   {mark} {f["metrics"].get("date", "")[11:19]}  {f["group"]:<24} {", ".join(f["reasons"])}{tag}')
    nok = sum(f['tier'] == 'ok' for f in frames)
    print(f'\n  {nok}/{len(frames)} frames ok ({nok / len(frames):.0%}).')


def apply_moves(store, frames, ask=True, dry_run=False):
    """Make the filesystem match decisions: rejects out to reject_dir, keeps back in.
    Files are only ever moved (never deleted); every move is logged and reversible."""
    to_move, to_restore = plan_moves(store, frames)
    if not to_move and not to_restore:
        print('  nothing to move.')
        return
    if dry_run:
        print(f'\n  DRY RUN - would move {len(to_move)} and restore {len(to_restore)} (nothing touched):')
        for f in to_move:
            print(f'   -> {f["path"]}\n      {store.planned_reject_path(f["path"])}   '
                  f'[{", ".join(f["reasons"]) or "manual"}]')
        for f in to_restore:
            print(f'   <- {f["path"]}\n      {store.moved.get(f["name"])}')
        return
    msg = f'  Move {len(to_move)} frame(s) to {store.reject_dir}' + \
          (f' and restore {len(to_restore)}' if to_restore else '') + '? [y/N] '
    if ask and input(msg).strip().lower() not in ('y', 'yes'):
        print('  left files untouched (decisions are saved; run blink/apply again later).')
        return
    errors = execute_moves(store, to_move, to_restore)
    for name, e in errors:
        print(f'  ! {name}: {e}')
    print(f'  moved {len(to_move) + len(to_restore) - len(errors)} file(s).')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmds = ('scan', 'blink', 'watch', 'restore', 'apply')
    if argv and argv[0] not in cmds and not argv[0].startswith('-'):
        argv = ['all'] + argv
    ap = argparse.ArgumentParser(prog='nightsift', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=cmds + ('all',))
    ap.add_argument('path')
    ap.add_argument('-j', '--workers', type=int, default=DEFAULT_WORKERS,
                    help=f'parallel workers (default {DEFAULT_WORKERS}; the disk is usually the limit, not CPU)')
    ap.add_argument('-s', '--sensitivity', '--strictness', type=float, default=1.0, dest='sensitivity',
                    help='>1 flags more aggressively, <1 more leniently (default 1.0)')
    ap.add_argument('-r', '--reject-dir', help='where rejected frames go (default: <project>_rejected); '
                    'remembered for this project')
    ap.add_argument('--osc', action=argparse.BooleanOptionalAction, default=None,
                    help='colour (OSC/Bayer) camera data; remembered for this project')
    ap.add_argument('--rescan', action='store_true',
                    help='scan: re-measure every frame and rebuild all previews (keeps your decisions)')
    ap.add_argument('--apply', action='store_true', help='scan/watch: move auto-rejects without asking')
    ap.add_argument('-n', '--dry-run', action='store_true', help='show what would be moved; touch nothing')
    ap.add_argument('--view', choices=('all', 'flagged', 'suspect', 'kept', 'rejected'), default='all',
                    help='blink: which frames to show (flagged = rejects + suspects)')
    ap.add_argument('--flagged', action='store_const', dest='view', const='flagged', help='same as --view flagged')
    ap.add_argument('--suspect', action='store_const', dest='view', const='suspect', help='same as --view suspect')
    ap.add_argument('--interval', type=float, default=15, help='watch: poll interval seconds')
    a = ap.parse_args(argv)

    store = Store(a.path, a.reject_dir)
    if a.osc is not None:
        store.config['osc'] = a.osc
    store.save()
    print(f'  project: {store.root}')
    print(f'  rejects: {store.reject_dir}')
    t0 = time.perf_counter()
    if a.cmd == 'restore':
        n = 0
        for p in store.discover(include_rejected=True):
            if store.in_reject_dir(p):
                if a.dry_run:
                    print(f'   would restore {p}\n      -> {store.moved.get(os.path.basename(p))}')
                    n += 1
                    continue
                store.restore(p)   # decisions are kept: restore only undoes the file move
                n += 1
        store.save()
        print(f'{"would restore" if a.dry_run else "restored"} {n} frame(s)')
        if n and not a.dry_run:
            print('  (they stay marked as rejects; to start fresh also delete .nightsift/decisions.json)')
        return

    if a.cmd == 'watch':
        print(f'watching {store.root} (Ctrl+C to stop)')
        try:
            while True:
                n = scan(store, a.workers, quiet=True)
                if n:
                    frames, stats = build_frames(store, a.sensitivity)
                    nr = sum(f['tier'] == 'reject' for f in frames)
                    ns = sum(f['tier'] == 'suspect' for f in frames)
                    print(f'{time.strftime("%H:%M:%S")}  +{n} scanned, {len(frames)} total, '
                          f'{nr} reject, {ns} suspect')
                    write_candidates(store, frames)
                    if a.apply or a.dry_run:
                        apply_moves(store, frames, ask=False, dry_run=a.dry_run)
                time.sleep(a.interval)
        except KeyboardInterrupt:
            return

    if a.cmd in ('scan', 'all'):
        n = scan(store, a.workers, force=a.rescan)
        print(f'  scanned {n} {"" if a.rescan else "new "}frame(s) in {time.perf_counter() - t0:.1f}s '
              f'({len(store.metrics)} cached)')
    frames, stats = build_frames(store, a.sensitivity)
    if not frames:
        print('no light frames found (run scan first?)')
        return
    if a.cmd in ('scan', 'all'):
        report(frames, stats, store)
        print(f'  list of flagged files: {write_candidates(store, frames)}')
    if a.cmd == 'scan':
        if a.apply or a.dry_run:
            apply_moves(store, frames, ask=False, dry_run=a.dry_run)
        else:
            print('  nothing moved (scan only). Check them with: nightsift blink --suspect <path>, '
                  'or move with --apply')
    if a.cmd in ('blink', 'all'):
        from .viewer import Blinker
        Blinker(store, frames, view=a.view).run()
        store.save()
        write_candidates(store, frames)
    if a.cmd in ('blink', 'all', 'apply'):
        apply_moves(store, frames, ask=True, dry_run=a.dry_run)


if __name__ == '__main__':
    main()
