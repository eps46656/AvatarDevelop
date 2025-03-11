
import abc
import torch
import dataclasses


@dataclasses.dataclass
class AvatarModel:
    poses: object
    vertices: torch.Tensor  # [..., V, 3]
    joint_rs: torch.Tensor  # [..., J, 3, 3]
    joint_ts: torch.Tensor  # [..., J, 3]


@abc.ABC
class AvatarBuilder(torch.nn.Model):
    @abc.abstractmethod
    def GetVerticesCnt(self) -> int:  # V
        pass

    @abc.abstractmethod
    def GetFacesCnt(self) -> int:  # F
        pass

    @abc.abstractmethod
    def GetJointsCnt(self) -> int:  # J
        pass

    @abc.abstractmethod
    def GetVertexTexturesCnt(self) -> torch.Tensor:  # VT
        pass

    @abc.abstractmethod
    def GetVertices(self) -> torch.Tensor:  # [..., V, 3]
        pass

    @abc.abstractmethod
    def GetFaces(self) -> torch.Tensor:  # [..., F, 3]
        pass

    @abc.abstractmethod
    def GetVertexTextures(self) -> torch.Tensor:  # [..., VT, 2]
        pass

    @abc.abstractmethod
    def GetFaceTextures(self) -> torch.Tensor:  # [..., F, 2]
        pass

    @abc.abstractmethod
    def GetAvatarModel(self, pose) -> AvatarModel:
        pass
