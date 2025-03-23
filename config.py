import pathlib

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

# ---

SMPL_MODELS_DIR = DIR / "smpl_models"

SMPL_MALE_MODEL = \
    SMPL_MODELS_DIR / "basicmodel_m_lbs_10_207_0_v1.1.0.pkl"

SMPL_FEMALE_MODEL = \
    SMPL_MODELS_DIR / "basicmodel_f_lbs_10_207_0_v1.1.0.pkl"

SMPL_NEUTRAL_MODEL = \
    SMPL_MODELS_DIR / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"

# ---

SMPLX_MODELS_DIR = DIR / "models_smplx_v1_1/models/smplx"

SMPLX_MALE_MODEL = SMPLX_MODELS_DIR / "SMPLX_MALE.pkl"

SMPLX_FEMALE_MODEL = SMPLX_MODELS_DIR / "SMPLX_FEMALE.pkl"

SMPLX_NEUTRAL_MODEL = SMPLX_MODELS_DIR / "SMPLX_NEUTRAL.pkl"

# ---

PEOPLE_SNAPSHOT_DIR = DIR / "people_snapshot_public"
