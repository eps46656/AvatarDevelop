import copy
import typing

import torch
from beartype import beartype

from .. import utils
from .ModelData import ModelData


@beartype
class ModelBuilder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_model_data(self) -> ModelData:
        raise utils.UnimplementationError()

    @property
    def device(self) -> torch.device:
        raise utils.UnimplementationError()

    def to(self) -> typing.Self:
        raise utils.UnimplementationError()

    def forward(self) -> ModelData:
        raise utils.UnimplementationError()


@beartype
class StaticModelBuilder(ModelBuilder):
    def __init__(self, model_data: ModelData):
        super().__init__()
        self.model_data = model_data

    def get_model_data(self) -> ModelData:
        return self.model_data

    @property
    def device(self) -> torch.device:
        return self.model_data.device

    def to(self, *args, **kwargs) -> typing.Self:
        self.model_data = self.model_data.to(*args, **kwargs)
        return self

    def forward(self) -> ModelData:
        return self.model_data


@beartype
class DeformableModelBuilder(ModelBuilder):
    def __init__(self, model_data: ModelData):
        super().__init__()
        self.model_data = copy.copy(model_data)

        self.model_data.vertex_positions = torch.nn.Parameter(
            self.model_data.vertex_positions.to(
                dtype=torch.float64, copy=True,
            ))

        self.register_parameter(
            "vertex_positions", self.model_data.vertex_positions)

    def get_model_data(self) -> ModelData:
        return self.model_data

    @property
    def device(self) -> torch.device:
        return self.model_data.device

    def to(self, *args, **kwargs) -> typing.Self:
        self.model_data = self.model_data.to(*args, **kwargs)
        return self

    def freeze(self):
        self.model_data.vertex_positions.requires_grad = False

    def unfreeze(self):
        self.model_data.vertex_positions.requires_grad = True

    def get_param_groups(self, base_lr: float):
        ret = list()

        if self.model_data.vertex_positions.requires_grad:
            ret.append({
                "params": [self.model_data.vertex_positions],
                "lr": min(1e-7, base_lr * 1e-2),
            })

        return ret

    def forward(self) -> ModelData:
        return self.model_data
