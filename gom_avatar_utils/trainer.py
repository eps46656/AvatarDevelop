import os
import torch
from beartype import beartype

from .. import smplx_utils, config
from . import model


@beartype
class Trainer:
    def __init__(
        self,
        *,
        smplx_model_data_path: os.PathLike,
        smplx_body_shapes_cnt: int,
        smplx_expr_shapes_cnt: int,
        smplx_body_joints_cnt: int,
        smplx_jaw_joints_cnt: int,
        smplx_eye_joints_cnt: int,
        smplx_hand_joints_cnt: int,
        device: torch.device,
    ):
        smplx_model_data = smplx_utils.ReadSMPLXModelData(
            model_data_path=smplx_model_data_path,
            body_shapes_cnt=smplx_body_shapes_cnt,
            expr_shapes_cnt=smplx_expr_shapes_cnt,
            body_joints_cnt=smplx_body_joints_cnt,
            jaw_joints_cnt=smplx_jaw_joints_cnt,
            eye_joints_cnt=smplx_eye_joints_cnt,
            hand_joints_cnt=smplx_hand_joints_cnt,
            device=device,
        )

        smplx_model_data.vertex_positions = torch.nn.Parameter(
            smplx_model_data.vertex_positions)

        self.smplx_model_builder = smplx_utils.SMPLXModelBuilder(
            model_data=smplx_model_data,
            device=device,
        )

        self.gom_avatar_model = model.GoMAvatarModel(
            avatar_blending_layer=self.smplx_model_builder,
            color_channels_cnt=3,
        )

    def forward(
        self,
        smplx_blending_param: smplx_utils.SMPLXBlendingParam,
    ):
        pass
