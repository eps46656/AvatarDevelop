import dataclasses

import torch
from beartype import beartype

from .. import utils
from .ModelData import ModelData


@beartype
class ModelBuilder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def GetModelData(self) -> ModelData:
        raise utils.UnimplementationError()

    def forward(self) -> ModelData:
        raise utils.UnimplementationError()


@beartype
class StaticModelBuilder(ModelBuilder):
    def __init__(self, model_data: ModelData):
        super().__init__()
        self.model_data = model_data

    def GetModelData(self) -> ModelData:
        return self.model_data

    def forward(self) -> ModelData:
        return self.model_data


@beartype
class DeformableModelBuilder(ModelBuilder):
    def __init__(self, model_data: ModelData):
        super().__init__()
        self.model_data = dataclasses.replace(model_data)

        self.model_data.vertex_positions = torch.nn.Parameter(
            self.model_data.vertex_positions)

    def GetModelData(self) -> ModelData:
        return self.model_data

    def forward(self) -> ModelData:
        return self.model_data
