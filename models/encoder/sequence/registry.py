from typing import Dict, Type

_SEQ_ENCODER_REGISTRY = {}


def register_sequence_encoder(name: str):
    def decorator(cls):
        _SEQ_ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def create_sequence_encoder(mode: str, **kwargs):
    if mode not in _SEQ_ENCODER_REGISTRY:
        raise ValueError(f"Unknown epoch factor '{mode}'. " f"Available: {list(_SEQ_ENCODER_REGISTRY.keys())}")
    return _SEQ_ENCODER_REGISTRY[mode](**kwargs)
