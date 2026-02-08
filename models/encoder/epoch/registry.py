from typing import Dict, Type

from .base import BaseEpochEncoder

_EPOCH_ENCODER_REGISTRY: Dict[str, Type[BaseEpochEncoder]] = {}


def register_epoch_encoder(name: str):
    def decorator(cls):
        _EPOCH_ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def create_epoch_encoder(mode: str, **kwargs) -> BaseEpochEncoder:
    if mode not in _EPOCH_ENCODER_REGISTRY:
        raise ValueError(f"Unknown epoch factor '{mode}'. " f"Available: {list(_EPOCH_ENCODER_REGISTRY.keys())}")
    return _EPOCH_ENCODER_REGISTRY[mode](**kwargs)
