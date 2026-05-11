"""Model builders for DeepPCB experiments."""

from collections.abc import Callable

from anomalib.models import Padim, Patchcore, get_model
from anomalib.models.components import AnomalibModule
from torch import nn

from .config import ModelConfig

ModelBuilder = Callable[[ModelConfig], AnomalibModule]


def _configure_pre_processor(model: AnomalibModule, config: ModelConfig) -> None:
    if hasattr(model, "configure_pre_processor"):
        model.pre_processor = model.configure_pre_processor(
            image_size=config.image_size,
            center_crop_size=config.center_crop_size,
        )


def build_patchcore(config: ModelConfig) -> Patchcore:
    """Build the PatchCore baseline."""

    model = Patchcore(
        backbone=config.backbone,
        pre_trained=config.pre_trained,
        layers=config.layers,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        num_neighbors=config.num_neighbors,
        **config.init_args,
    )
    _configure_pre_processor(model, config)
    return model


def build_patchcore_simam(config: ModelConfig) -> Patchcore:
    """Build PatchCore + SimAM with local max-pooling for PCB defects."""

    model = Patchcore(
        backbone=config.backbone,
        pre_trained=config.pre_trained,
        layers=config.layers,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        num_neighbors=config.num_neighbors,
        use_simam=True,
        **config.init_args,
    )
    model.model.feature_pooler = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    _configure_pre_processor(model, config)
    return model


def build_padim(config: ModelConfig) -> Padim:
    """Build the PaDiM baseline."""

    model = Padim(
        backbone=config.backbone,
        pre_trained=config.pre_trained,
        layers=config.layers,
        n_features=config.n_features,
        **config.init_args,
    )
    return model


MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "patchcore": build_patchcore,
    "patchcore_simam": build_patchcore_simam,
    "padim": build_padim,
}


def build_model(config: ModelConfig) -> AnomalibModule:
    """Build a configured Anomalib model.

    Known PCB variants use explicit builders. Any other Anomalib model name is
    passed to ``get_model`` with ``init_args`` so new experiments can be added
    without editing this function first.
    """

    if config.name in MODEL_BUILDERS:
        return MODEL_BUILDERS[config.name](config)

    return get_model(config.name, **config.init_args)

