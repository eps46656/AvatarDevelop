import enum

from .blending_utils import *
from .config import *
from .Model import *
from .ModelBlender import *
from .ModelBuilder import *
from .ModelConfig import *
from .ModelData import *


class Gender(enum.Enum):
    NEUTRAL = enum.auto()
    MALE = enum.auto()
    FEMALE = enum.auto()
