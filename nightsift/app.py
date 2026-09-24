"""Launcher for the desktop app:  python -m nightsift.app [project folder]

Kept tiny on purpose: scan workers are separate processes that re-import this module,
so it must not import Qt at top level.
"""
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from nightsift.gui import main
    main()
