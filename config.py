import pathlib

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

# ---

TMP_DIR = DIR / "tmp"

# ---

SMPL_MODELS_DIR = DIR / "smpl_models"

SMPL_MALE_MODEL_PATH = \
    SMPL_MODELS_DIR / "basicmodel_m_lbs_10_207_0_v1.1.0.pkl"

SMPL_FEMALE_MODEL_PATH = \
    SMPL_MODELS_DIR / "basicmodel_f_lbs_10_207_0_v1.1.0.pkl"

SMPL_NEUTRAL_MODEL_PATH = \
    SMPL_MODELS_DIR / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"

# ---

SMPLX_MODELS_DIR = DIR / "models_smplx_v1_1/models/smplx"

SMPLX_MALE_MODEL_PATH = SMPLX_MODELS_DIR / "SMPLX_MALE.pkl"

SMPLX_FEMALE_MODEL_PATH = SMPLX_MODELS_DIR / "SMPLX_FEMALE.pkl"

SMPLX_NEUTRAL_MODEL_PATH = SMPLX_MODELS_DIR / "SMPLX_NEUTRAL.pkl"

# ---

PEOPLE_SNAPSHOT_DIR = DIR / "people_snapshot_public"

# ---

CLOTH3D_DIR = DIR / "cloth3d"

# ---

DENSEPOSE_CONFIG = DIR / "densepose_data/densepose_rcnn_R_50_FPN_s1x.yaml"

DENSEPOSE_MODEL = DIR / "densepose_data/model_final_162be9.pkl"

# ---

OPENPOSE_DIR = DIR / "openpose"

# OPENPOSE_BIN =

# ---

DD_HUMAN_PARSING_DIR = DIR / "2D-Human-Parsing"

# ---

M3D_VTON_DIR = DIR / "M3D-VTON"

# ---

GART_DIR = DIR / "GART"
