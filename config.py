import torch
import enum


class Empty:
    pass


class DType(enum.Enum):
    FLOAT = torch.float32
    LONG = torch.long


class DType(enum.Enum):
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
