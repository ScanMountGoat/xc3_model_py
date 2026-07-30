from typing import ClassVar

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
    morph_tracks: MorphTracks | None
    root_translation: numpy.ndarray | None

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
    ) -> tuple[float, float, float] | None: ...
    def sample_rotation(
        self, frame: float, frame_count: int
    ) -> tuple[float, float, float] | None: ...
    def sample_scale(
        self, frame: float, frame_count: int
    ) -> tuple[float, float, float] | None: ...
    def sample_transform(self, frame: float) -> Transform | None: ...
    def bone_index(self) -> int | None: ...
    def bone_hash(self) -> int | None: ...
    def bone_name(self) -> str | None: ...

class MorphTracks:
    track_indices: numpy.ndarray
    track_values: numpy.ndarray

class KeyFrame:
    x_coeffs: tuple[float, float, float, float]
    y_coeffs: tuple[float, float, float, float]
    z_coeffs: tuple[float, float, float, float]
    w_coeffs: tuple[float, float, float, float]

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
