"""Command line entry point for DeepPCB experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcb_defect.core.config import EXPERIMENTS, get_experiment_config
from pcb_defect.core.prepare import prepare_deep_pcb_folder
from pcb_defect.core.runner import run_test, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepPCB Anomalib experiments.")
    parser.add_argument(
        "command",
        choices=["prepare", "train", "test", "train-test"],
        help="Action to run.",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        choices=sorted(EXPERIMENTS),
        default="patchcore_simam",
        help="Experiment configuration to use.",
    )
    parser.add_argument("--ckpt-path", type=Path, default=None, help="Checkpoint path for test.")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "prepare":
        prepare_deep_pcb_folder()
        return

    config = get_experiment_config(args.experiment)
    if args.train_batch_size is not None:
        config.data.train_batch_size = args.train_batch_size
    if args.eval_batch_size is not None:
        config.data.eval_batch_size = args.eval_batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers

    if args.command == "train":
        run_train(config)
    elif args.command == "train-test":
        run_train(config, test_after_fit=True)
    else:
        run_test(config, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
