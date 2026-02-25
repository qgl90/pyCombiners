"""Allow running the package with `python -m trackcomb`."""
__author__ = "Renato Quagliani <rquaglia@cern.ch>"


from .cli import main

raise SystemExit(main())
