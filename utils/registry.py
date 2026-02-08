# fmt: off

# Trainer and model class path registries
TRAINER_REGISTRY = {
    "base"  : "trainers.base_trainer.BaseTrainer",
    
}

MODEL_REGISTRY = {
    "sleep_base" : "models.sleep_base.SleepBase",
    "sleep_band" : "models.sleep_band.SleepBand",
   
}

METHOD_REGISTRY = {
    "cmmd":     "methods.cmmd.ClassConditionalMMD",
    "coral":    "methods.coral.CORAL",
    "dann":     "methods.dann.DANN",
    "irm":      "methods.irm.IRM",
    "mmd":      "methods.mmd.MMD",
    "relcoral": "methods.relcoral.RelCORAL",
}

FRONTEND_REGISTRY = {
    "sinc":                 "frontend.sinc.SincConv1d",
    "gabor":                "frontend.gabor.GaborConv1d",
    "constant_q":           "frontend.gabor.ConstantQGaborConv1d",
    "grouped_constant_q":   "frontend.gabor.GroupedConstantQMorletConv1d",
}
# fmt: on


def _get_class_by_path(path: str):
    """Dynamically import a class from a string path."""
    import importlib

    try:
        module_name, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import '{path}': {e}") from e


def get_trainer(name: str):
    """Get trainer class by name."""
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"❌ Trainer '{name}' not found. Available: {list(TRAINER_REGISTRY.keys())}")
    return _get_class_by_path(TRAINER_REGISTRY[name])


def get_model(name: str):
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"❌ Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return _get_class_by_path(MODEL_REGISTRY[name])
