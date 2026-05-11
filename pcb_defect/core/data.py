"""DeepPCB data preparation and datamodule construction."""

from anomalib.data import Folder

from .config import DataConfig
from .prepare import prepare_deep_pcb_folder


def build_datamodule(config: DataConfig) -> Folder:
    """Build the Anomalib Folder datamodule for DeepPCB."""

    return Folder(
        name=config.name,
        root=str(config.root),
        normal_dir=config.normal_dir,
        abnormal_dir=config.abnormal_dir,
        normal_test_dir=config.normal_test_dir,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        test_split_mode=config.test_split_mode,
        val_split_mode=config.val_split_mode,
        val_split_ratio=config.val_split_ratio,
        seed=config.seed,
    )


__all__ = ["build_datamodule", "prepare_deep_pcb_folder"]

