"""
VGA Models Package.
Contains the Vision Projector, Ray-RoPE 3D Module, Action Expert DiT, and VGA Policy.
"""

from models.projector import UnifiedSpaceToDepthProjector
from models.ray_rope import CentroidRayRoPE

__all__ = ["UnifiedSpaceToDepthProjector", "CentroidRayRoPE"]
