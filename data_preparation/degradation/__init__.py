"""RealBasicVSR-style video degradation (second-order Real-ESRGAN degradation per frame
followed by video compression), adapted from Upscale-A-Video / STAR data generation."""
from .degrade_video import DIFFICULTY_PRESETS, degradation_process

__all__ = ["DIFFICULTY_PRESETS", "degradation_process"]
