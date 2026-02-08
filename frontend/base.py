from abc import ABC
from torch import nn


class BaseFrontend(nn.Module, ABC):
    """The frontend class is used to filter raw input signals."""

    out_channels: int
