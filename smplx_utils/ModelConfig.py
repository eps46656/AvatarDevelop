import dataclasses

from beartype import beartype


@beartype
@dataclasses.dataclass
class ModelConfig:
    body_shapes_cnt: int
    expr_shapes_cnt: int

    body_joints_cnt: int
    jaw_joints_cnt: int
    eye_joints_cnt: int
    hand_joints_cnt: int


smpl_model_config = ModelConfig(
    body_shapes_cnt=10,
    expr_shapes_cnt=0,

    body_joints_cnt=24,
    jaw_joints_cnt=0,
    eye_joints_cnt=0,
    hand_joints_cnt=0,
)

smplx_model_config = ModelConfig(
    body_shapes_cnt=10,
    expr_shapes_cnt=10,

    body_joints_cnt=22,
    jaw_joints_cnt=1,
    eye_joints_cnt=1,
    hand_joints_cnt=15,
)
