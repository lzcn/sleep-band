from typing import Dict, Type

from .base import DomainMethod

_DOMAIN_METHOD_REGISTRY: Dict[str, Type[DomainMethod]] = {}


def register_domain_method(name: str):
    def decorator(cls):
        _DOMAIN_METHOD_REGISTRY[name] = cls
        return cls

    return decorator


def create_domain_method(mode: str, **kwargs) -> DomainMethod:
    if mode not in _DOMAIN_METHOD_REGISTRY:
        raise ValueError(f"Unknown epoch factor '{mode}'. " f"Available: {list(_DOMAIN_METHOD_REGISTRY.keys())}")
    return _DOMAIN_METHOD_REGISTRY[mode](**kwargs)
