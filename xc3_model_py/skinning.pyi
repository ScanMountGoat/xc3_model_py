from typing import Optional, ClassVar, Tuple
import numpy
from xc3_model_py.material import RenderPassType

class Skinning:
    bones: list[Bone]

    def __init__(self, bones: list[Bone]) -> None: ...

class Bone:
    name: str
    bounds: Optional[BoneBounds]
    constraint: Optional[BoneConstraint]
    no_camera_overlap: bool

    def __init__(
        self,
        name: str,
        no_camera_overlap: bool,
        bounds: Optional[BoneBounds],
        constraint: Optional[BoneConstraint],
    ) -> None: ...

class BoneBounds:
    center: list[float]
    size: list[float]
    radius: float

    def __init__(
        self,
        center: list[float],
        size: list[float],
        radius: float,
    ) -> None: ...

class BoneConstraint:
    fixed_offset: list[float]
    max_distance: float
    constraint_type: BoneConstraintType
    parent_index: Optional[int]

    def __init__(
        self,
        fixed_offset: list[float],
        max_distance: float,
        constraint_type: BoneConstraintType,
        parent_index: Optional[int],
    ) -> None: ...

class BoneConstraintType:
    FixedOffset: ClassVar[BoneConstraintType]
    Distance: ClassVar[BoneConstraintType]

class Weights:
    weight_buffers: list[SkinWeights]

    def __init__(self, weight_buffers: list[SkinWeights]) -> None: ...
    def weight_buffer(self, flags2: int) -> Optional[SkinWeights]: ...
    def weights_start_index(
        self, flags2: int, lod_item_index: int, unk_type: RenderPassType
    ) -> int: ...
    def update_weights(self, combined_weights: SkinWeights) -> None: ...

class SkinWeights:
    bone_indices: numpy.ndarray
    weights: numpy.ndarray
    bone_names: list[str]

    def __init__(
        self, bone_indices: numpy.ndarray, weights: numpy.ndarray, bone_names: list[str]
    ) -> None: ...
    def to_influences(self, weight_indices: numpy.ndarray) -> list[Influence]: ...
    def add_influences(self, influences: list[Influence]) -> numpy.ndarray: ...

class Influence:
    bone_name: str
    weights: list[VertexWeight]

    def __init__(self, bone_name: str, weights: list[VertexWeight]) -> None: ...

class VertexWeight:
    vertex_index: int
    weight: float

    def __init__(self, vertex_index: int, weight: float) -> None: ...
