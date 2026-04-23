"""Thin entry point: delegate to the grid's shared training loop."""
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from _shared_train import main

if __name__ == '__main__':
    main()
