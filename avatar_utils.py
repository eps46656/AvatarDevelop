
import torch


class AvatarModel:
    def GetVertices(self) -> torch.Tensor:  # [..., V, 3]
        assert False, "Unoverridden abstract method."


class AvatarBuilder(torch.nn.Model):
    def GetVerticesCnt(self) -> int:  # V
        assert False, "Unoverridden abstract method."

    def GetFacesCnt(self) -> int:  # F
        assert False, "Unoverridden abstract method."

    def GetJointsCnt(self) -> int:  # J
        assert False, "Unoverridden abstract method."

    def GetVertexTexturesCnt(self) -> torch.Tensor:  # VT
        assert False, "Unoverridden abstract method."

    def GetVertices(self) -> torch.Tensor:  # [..., V, 3]
        assert False, "Unoverridden abstract method."

    def GetFaces(self) -> torch.Tensor:  # [..., F, 3]
        assert False, "Unoverridden abstract method."

    def GetVertexTextures(self) -> torch.Tensor:  # [..., VT, 2]
        assert False, "Unoverridden abstract method."

    def GetFaceTextures(self) -> torch.Tensor:  # [..., F, 2]
        assert False, "Unoverridden abstract method."

    def forward(self, blending_param) -> AvatarModel:
        assert False, "Unoverridden abstract method."
