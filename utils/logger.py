import io
import logging
from logging.config import dictConfig
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

try:
    import swanlab
except ImportError:
    swanlab = None

NAMED_FORMATTERS = {
    "default": {
        "format": "[%(levelname)s] - %(asctime)s - [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
        "datefmt": "%m-%d %H:%M:%S",
    },
    "simple": {
        "format": "[%(levelname)s] - %(asctime)s - [%(name)s]: %(message)s",
        "datefmt": "%m-%d %H:%M:%S",
    },
    "concise": {
        "format": "%(asctime)s: %(message)s",
        "datefmt": "%m-%d %H:%M:%S",
    },
}


def get_value(x, default):
    return default if x is None else x


def config_logger(
    level: str = "INFO",
    stream_level: Optional[str] = None,
    file_level: Optional[str] = None,
    log_file: Optional[str] = None,
    file_mode: str = "a",
    formatter: str = "default",
    file_formatter: Optional[str] = None,
    stream_formatter: Optional[str] = None,
):
    """Minimal logger config, no colors."""
    file_level = get_value(file_level, level)
    file_formatter = get_value(file_formatter, formatter)
    stream_level = get_value(stream_level, level)
    stream_formatter = get_value(stream_formatter, formatter)

    formatters = NAMED_FORMATTERS

    stream_handler = {
        "class": "logging.StreamHandler",
        "formatter": stream_formatter,
        "level": stream_level,
    }

    file_handler = {
        "class": "logging.FileHandler",
        "formatter": file_formatter,
        "level": file_level,
        "filename": log_file,
        "mode": file_mode,
    }

    handlers = {"stream": stream_handler}
    if log_file:
        handlers["file"] = file_handler

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "root": {"level": "DEBUG", "handlers": list(handlers.keys())},
        }
    )


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


class WriterAdapter:
    def __init__(
        self,
        log_dir: str,
        project_name: str = "default_project",
        run_name: str = "default_run",
        config: Optional[Dict[str, Any]] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.tensorboard = SummaryWriter(log_dir=str(self.log_dir)) if use_tensorboard else None
        self.swanlab = None

        if use_wandb and swanlab:
            # close any existing SwanLab instance
            if swanlab.run is not None:
                swanlab.finish()
            self.swanlab = swanlab.init(
                project=project_name,
                experiment_name=run_name,
                config=config or {},
            )

    @classmethod
    def from_params(cls, params: Any, log_dir: str) -> "WriterAdapter":
        return cls(
            log_dir=log_dir,
            project_name=getattr(params, "project_name", "default_project"),
            run_name=getattr(params, "run_name", "default_run"),
            config=vars(params) if hasattr(params, "__dict__") else {},
            use_tensorboard=True,
            use_wandb=getattr(params, "use_wandb", False),
        )

    def log_metrics(self, data: Dict[str, float], step: int):
        if self.tensorboard:
            for k, v in data.items():
                self.tensorboard.add_scalar(k, v, step)
        if self.swanlab:
            self.swanlab.log(data, step=step)

    def log_image_tensor(self, tag: str, image_tensor, step: int):
        if self.tensorboard:
            self.tensorboard.add_image(tag, image_tensor, step)
        if self.swanlab:
            self.swanlab.log({tag: swanlab.Image(image_tensor)}, step=step)

    def log_confusion_matrix(
        self,
        cm,
        class_names,
        tag: str = "ConfusionMatrix",
        step: int = 0,
        normalize: bool = True,
    ):
        """Logs a confusion matrix as an image to TensorBoard or SwanLab.

        Args:
            writer: The logging object (e.g., TensorBoard SummaryWriter or SwanLab logger).
            cm (ndarray): Confusion matrix (2D array).
            class_names (list): List of class names corresponding to the confusion matrix.
            tag (str): Tag for the image in the logger. Default is "ConfusionMatrix".
            step (int): The global step value to record. Default is 0.
            normalize (bool): Whether to normalize the confusion matrix. Default is True.

        Returns:
            None
        """
        # Normalize the confusion matrix if required
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cm = cm.astype("float") / row_sums
            fmt = ".2f"
            vmin, vmax = 0, 1
        else:
            fmt = "d"
            vmin, vmax = None, None

        # Create a heatmap visualization of the confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(tag)
        plt.tight_layout()

        # Convert the plot to an image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        image_tensor = transforms.ToTensor()(image)

        # Log the image to the writer
        if self.tensorboard:
            self.tensorboard.add_image(tag, image_tensor, step)
        if self.swanlab:
            self.swanlab.log({tag: swanlab.Image(image_tensor)}, step=step)

        # Close the plot to free resources
        plt.close(fig)
