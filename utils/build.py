from __future__ import annotations

from typing import Any, Dict, Mapping
from torch import nn
from frontend import BaseFrontend
from methods import DomainMethod
from trainers import BaseTrainer


def _import_from_string(path: str):
    """Dynamically import a class from a string path."""
    import importlib

    try:
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import '{path}': {e}") from e


def _resolve_registry_entry(entry: Any):
    """
    Registry entry can be:
        - class
        - callable
        - string path
    """
    if isinstance(entry, str):
        return _import_from_string(entry)
    return entry


def build_from_cfg(cfg: dict | None, registry: dict, *, name_key: str = "name", **kwargs):
    if cfg is None:
        return None

    cfg = dict(cfg)

    name = cfg.pop(name_key)

    if name not in registry:
        raise KeyError(f"{name} not in registry")

    cls = _resolve_registry_entry(registry[name])

    cfg.update(kwargs)

    return cls(**cfg)


def build_model(cfg, **kwargs) -> nn.Module:
    from .registry import MODEL_REGISTRY

    return build_from_cfg(cfg, MODEL_REGISTRY, **kwargs)


def build_trainer(cfg, **kwargs) -> BaseTrainer:
    from .registry import TRAINER_REGISTRY

    return build_from_cfg(cfg, TRAINER_REGISTRY, **kwargs)


def build_domain_method(cfg, **kwargs) -> DomainMethod:
    from .registry import METHOD_REGISTRY

    return build_from_cfg(cfg, METHOD_REGISTRY, **kwargs)


def build_frontend(cfg, **kwargs) -> BaseFrontend:
    from .registry import FRONTEND_REGISTRY

    return build_from_cfg(cfg, FRONTEND_REGISTRY, **kwargs)
