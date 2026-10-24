"""Run a script with torch imported before lightgbm (the reverse order segfaults on macOS: duplicate libomp)."""
import runpy
import sys

import torch  # noqa: F401

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
