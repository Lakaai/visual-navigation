from enum import Enum

class UpdateMethod(Enum):
    AFFINE = 1
    UNSCENTED = 2
    BFGS = 3
    NEWTON = 4