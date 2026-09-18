"""Fixed, non-trainable boundaries between Balatro and the fly."""

from flylatro.interface.motor import FixedMotorInterface, MotorMapping
from flylatro.interface.sensory import FixedPlasticSensoryEncoder, SensoryMapping

__all__ = [
    "FixedMotorInterface",
    "FixedPlasticSensoryEncoder",
    "MotorMapping",
    "SensoryMapping",
]
