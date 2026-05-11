"""Train/test runners shared by DeepPCB scripts and CLI."""

from pathlib import Path

from anomalib.engine import Engine
from lightning.pytorch.loggers import CSVLogger

from .config import ExperimentConfig
from .data import build_datamodule
from .models import build_model
from .prepare import prepare_deep_pcb_folder


def build_engine(config: ExperimentConfig, logger_name: str | None = None) -> Engine:
    """Build an Anomalib engine with a CSV logger."""

    log_name = logger_name or config.trainer.logger_name
    logger = CSVLogger(save_dir=str(config.trainer.default_root_dir), name=log_name)
    return Engine(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        default_root_dir=str(config.trainer.default_root_dir),
        logger=logger,
    )


def run_train(config: ExperimentConfig, test_after_fit: bool = False) -> None:
    """Prepare data, fit a model, and optionally test it."""

    if config.prepare_data:
        prepare_deep_pcb_folder()
    config.trainer.default_root_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(config.data)
    model = build_model(config.model)
    engine = build_engine(config)

    print(f"=== FIT {config.model.name} on DeepPCB folder dataset ===")
    print(f"Data root: {config.data.root}")
    engine.fit(model=model, datamodule=datamodule)

    if test_after_fit:
        print(f"=== TEST {config.model.name} on DeepPCB folder dataset ===")
        engine.test(model=model, datamodule=datamodule)

    print("=== COMPLETE ===")
    print(f"Results and logs are saved in: {config.trainer.default_root_dir}")


def find_latest_checkpoint(results_root: Path, logger_name: str) -> Path | None:
    """Find the newest Lightning checkpoint below a logger directory."""

    candidates = sorted(
        (results_root / logger_name).glob("version_*/checkpoints/*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    candidates = sorted(
        results_root.glob(f"**/{logger_name}/**/*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_test(config: ExperimentConfig, ckpt_path: Path | None = None) -> None:
    """Test a configured model from a checkpoint."""

    checkpoint = ckpt_path or find_latest_checkpoint(config.trainer.default_root_dir, config.trainer.logger_name)
    if checkpoint is None:
        print(f"Checkpoint not found for logger '{config.trainer.logger_name}'. Please train first.")
        return

    datamodule = build_datamodule(config.data)
    model = build_model(config.model)
    engine = build_engine(config, logger_name=f"{config.trainer.logger_name}_test")

    print(f"=== TEST {config.model.name} on DeepPCB folder dataset ===")
    print(f"Loading checkpoint from: {checkpoint}")
    engine.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint))

    print("=== TEST COMPLETE ===")
    print(f"Results and logs are saved in: {config.trainer.default_root_dir}")

