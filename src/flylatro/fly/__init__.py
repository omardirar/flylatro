"""Fixed encoders, fly backends, and neural feature extraction."""

from flylatro.fly.backend import FlyActivity, FlyBackend, TinyGraphFlyBackend
from flylatro.fly.encoder import EncoderSpec, FixedBalatroEncoder, Stimulus
from flylatro.fly.features import FeatureSpec, RateFeatureExtractor

__all__ = [
    "EncoderSpec",
    "FeatureSpec",
    "FixedBalatroEncoder",
    "FlyActivity",
    "FlyBackend",
    "RateFeatureExtractor",
    "Stimulus",
    "TinyGraphFlyBackend",
]

