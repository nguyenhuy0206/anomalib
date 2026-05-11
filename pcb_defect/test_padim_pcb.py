"""Test the PaDiM DeepPCB experiment."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcb_defect.core.config import get_experiment_config
from pcb_defect.core.runner import run_test


def main() -> None:
    config = get_experiment_config("padim")
    run_test(config)


if __name__ == "__main__":
    main()
