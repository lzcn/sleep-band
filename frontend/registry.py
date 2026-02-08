from typing import Dict, Type
from .base import BaseFrontend

_FRONTEND_REGISTRY: Dict[str, Type[BaseFrontend]] = {}


def register_frontend(name: str):
    def decorator(cls):
        _FRONTEND_REGISTRY[name] = cls
        return cls

    return decorator


def create_frontend(mode: str, **kwargs) -> BaseFrontend:
    if mode not in _FRONTEND_REGISTRY:
        raise ValueError(f"Unknown epoch factor '{mode}'. " f"Available: {list(_FRONTEND_REGISTRY.keys())}")
    return _FRONTEND_REGISTRY[mode](**kwargs)
