"""Configuration objects for DeepPCB experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
RESULTS_ROOT = PACKAGE_ROOT / "results"


@dataclass(slots=True)
class DataConfig:
    """Folder datamodule settings shared by all DeepPCB models."""

    name: str = "deep_pcb"
    root: Path = DATA_ROOT
    normal_dir: str = "train/normal"
    abnormal_dir: str = "test/abnormal"
    normal_test_dir: str = "test/normal"
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 0
    test_split_mode: str = "from_dir"
    val_split_mode: str = "same_as_test"
    val_split_ratio: float = 0.5
    seed: int = 42


@dataclass(slots=True)
class ModelConfig:
    """Model settings with defaults tuned for DeepPCB experiments."""

    name: str
    backbone: str = "resnet18"
    pre_trained: bool = True
    layers: list[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])
    image_size: tuple[int, int] = (256, 256)
    center_crop_size: tuple[int, int] | None = None
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    n_features: int | None = None
    init_args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainerConfig:
    """Engine and logger settings."""

    max_epochs: int = 1
    accelerator: str = "auto"
    devices: int = 1
    default_root_dir: Path = RESULTS_ROOT
    logger_name: str = "deep_pcb"


@dataclass(slots=True)
class ExperimentConfig:
    """Complete experiment description."""

    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    prepare_data: bool = True


def patchcore_config() -> ExperimentConfig:
    """PatchCore baseline config."""

    return ExperimentConfig(
        model=ModelConfig(name="patchcore"),
        trainer=TrainerConfig(logger_name="patchcore_pcb"),
    )


def patchcore_simam_config() -> ExperimentConfig:
    """PatchCore + SimAM config for small PCB defects."""

    return ExperimentConfig(
        model=ModelConfig(name="patchcore_simam"),
        data=DataConfig(name="deep_pcb_simam"),
        trainer=TrainerConfig(logger_name="patchcore_simam_pcb"),
    )


def padim_config() -> ExperimentConfig:
    """PaDiM baseline config."""

    return ExperimentConfig(
        model=ModelConfig(name="padim"),
        trainer=TrainerConfig(logger_name="padim_pcb"),
    )


EXPERIMENTS = {
    "patchcore": patchcore_config,
    "patchcore_simam": patchcore_simam_config,
    "padim": padim_config,
}


def get_experiment_config(name: str) -> ExperimentConfig:
    """Return a named experiment configuration."""

    try:
        return EXPERIMENTS[name]()
    except KeyError as error:
        available = ", ".join(sorted(EXPERIMENTS))
        msg = f"Unknown PCB experiment '{name}'. Available: {available}"
        raise ValueError(msg) from error
