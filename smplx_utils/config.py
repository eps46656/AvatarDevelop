from .. import transform_utils

BODY_SHAPES_SPACE_DIM = 300

smpl_model_transform = transform_utils.ObjectTransform.FromMatching("LUF")
smplx_model_transform = transform_utils.ObjectTransform.FromMatching("LUF")
