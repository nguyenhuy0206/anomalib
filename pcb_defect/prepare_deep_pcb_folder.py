"""Compatibility wrapper for preparing the DeepPCB folder dataset."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcb_defect.core.prepare import prepare_deep_pcb_folder


if __name__ == "__main__":
    prepare_deep_pcb_folder()
