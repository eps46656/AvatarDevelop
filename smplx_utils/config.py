from .. import transform_utils

BODY_SHAPES_SPACE_DIM = 300

smpl_model_transform = transform_utils.ObjectTransform.from_matching("LUF")
smplx_model_transform = transform_utils.ObjectTransform.from_matching("LUF")
