from typing import Optional, ClassVar
import numpy

from xc3_model_py.skinning import Weights

class ModelBuffers:
    vertex_buffers: list[VertexBuffer]
    index_buffers: list[IndexBuffer]
    weights: Optional[Weights]

    def __init__(
        self,
        vertex_buffers: list[VertexBuffer],
        index_buffers: list[IndexBuffer],
        weights: Optional[Weights],
    ) -> None: ...

class VertexBuffer:
    attributes: list[AttributeData]
    morph_blend_target: list[AttributeData]
    morph_targets: list[MorphTarget]
    outline_buffer_index: Optional[int]

    def __init__(
        self,
        attributes: list[AttributeData],
        morph_targets: list[MorphTarget],
        outline_buffer_index: Optional[int],
    ) -> None: ...

class AttributeData:
    attribute_type: AttributeType
    data: numpy.ndarray

    def __init__(self, attribute_type: AttributeType, data: numpy.ndarray) -> None: ...

class AttributeType:
    Position: ClassVar[AttributeType]
    Normal: ClassVar[AttributeType]
    Tangent: ClassVar[AttributeType]
    TexCoord0: ClassVar[AttributeType]
    TexCoord1: ClassVar[AttributeType]
    TexCoord2: ClassVar[AttributeType]
    TexCoord3: ClassVar[AttributeType]
    TexCoord4: ClassVar[AttributeType]
    TexCoord5: ClassVar[AttributeType]
    TexCoord6: ClassVar[AttributeType]
    TexCoord7: ClassVar[AttributeType]
    TexCoord8: ClassVar[AttributeType]
    VertexColor: ClassVar[AttributeType]
    Blend: ClassVar[AttributeType]
    WeightIndex: ClassVar[AttributeType]
    Position2: ClassVar[AttributeType]
    Normal4: ClassVar[AttributeType]
    OldPosition: ClassVar[AttributeType]
    Tangent2: ClassVar[AttributeType]
    SkinWeights: ClassVar[AttributeType]
    SkinWeights2: ClassVar[AttributeType]
    BoneIndices: ClassVar[AttributeType]
    BoneIndices2: ClassVar[AttributeType]

class MorphTarget:
    morph_controller_index: int
    position_deltas: numpy.ndarray
    normals: numpy.ndarray
    tangents: numpy.ndarray
    vertex_indices: numpy.ndarray

    def __init__(
        self,
        morph_controller_index: int,
        position_deltas: numpy.ndarray,
        normals: numpy.ndarray,
        tangents: numpy.ndarray,
        vertex_indices: numpy.ndarray,
    ) -> None: ...

class IndexBuffer:
    indices: numpy.ndarray

    def __init__(self, indices: numpy.ndarray) -> None: ...
