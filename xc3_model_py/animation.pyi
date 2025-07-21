from typing import Optional, ClassVar, Tuple
import numpy

from xc3_model_py import Skeleton, Transform

class Animation:
    name: str
    space_mode: SpaceMode
    play_mode: PlayMode
    blend_mode: BlendMode
    frames_per_second: float
    frame_count: int
    tracks: list[Track]
    root_translation: Optional[numpy.ndarray]

    def current_frame(self, current_time_seconds: float) -> float: ...
    def skinning_transforms(
        self, skeleton: Skeleton, frame: float
    ) -> numpy.ndarray: ...
    def model_space_transforms(
        self, skeleton: Skeleton, frame: float
    ) -> numpy.ndarray: ...
    def local_space_transforms(
        self, skeleton: Skeleton, frame: float
    ) -> numpy.ndarray: ...
    def fcurves(self, skeleton: Skeleton, use_blender_coordinates: bool) -> FCurves: ...

class Track:
    def sample_translation(
        self, frame: float, frame_count: int
    ) -> Optional[Tuple[float, float, float]]: ...
    def sample_rotation(
        self, frame: float, frame_count: int
    ) -> Optional[Tuple[float, float, float]]: ...
    def sample_scale(
        self, frame: float, frame_count: int
    ) -> Optional[Tuple[float, float, float]]: ...
    def sample_transform(self, frame: float) -> Optional[Transform]: ...
    def bone_index(self) -> Optional[int]: ...
    def bone_hash(self) -> Optional[int]: ...
    def bone_name(self) -> Optional[str]: ...

class KeyFrame:
    x_coeffs: Tuple[float, float, float, float]
    y_coeffs: Tuple[float, float, float, float]
    z_coeffs: Tuple[float, float, float, float]
    w_coeffs: Tuple[float, float, float, float]

class SpaceMode:
    Local: ClassVar[SpaceMode]
    Model: ClassVar[SpaceMode]

class PlayMode:
    Loop: ClassVar[PlayMode]
    Single: ClassVar[PlayMode]

class BlendMode:
    Blend: ClassVar[BlendMode]
    Add: ClassVar[BlendMode]

class FCurves:
    translation: dict[str, numpy.ndarray]
    rotation: dict[str, numpy.ndarray]
    scale: dict[str, numpy.ndarray]

def murmur3(name: str) -> int: ...
