from typing import Optional, ClassVar
import numpy

from xc3_model_py.skinning import Weights

class ModelBuffers:
    vertex_buffers: list[VertexBuffer]
    outline_buffers: list[OutlineBuffer]
    index_buffers: list[IndexBuffer]
    unk_buffers: list[UnkBuffer]
    unk_data: Optional[UnkDataBuffer]
    weights: Optional[Weights]

    def __init__(
        self,
        vertex_buffers: list[VertexBuffer],
        outline_buffers: list[OutlineBuffer],
        index_buffers: list[IndexBuffer],
        unk_buffers: list[UnkBuffer],
        unk_data: Optional[UnkDataBuffer],
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
    SkinWeights2: ClassVar[AttributeType]
    BoneIndices2: ClassVar[AttributeType]
    WeightIndex: ClassVar[AttributeType]
    WeightIndex2: ClassVar[AttributeType]
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
    Unk18: ClassVar[AttributeType]
    Blend: ClassVar[AttributeType]
    Unk15: ClassVar[AttributeType]
    Unk16: ClassVar[AttributeType]
    Normal: ClassVar[AttributeType]
    Tangent: ClassVar[AttributeType]
    Unk31: ClassVar[AttributeType]
    Normal2: ClassVar[AttributeType]
    ValInf: ClassVar[AttributeType]
    Position2: ClassVar[AttributeType]
    Normal4: ClassVar[AttributeType]
    OldPosition: ClassVar[AttributeType]
    Tangent2: ClassVar[AttributeType]
    SkinWeights: ClassVar[AttributeType]
    BoneIndices: ClassVar[AttributeType]

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
    primitive_type: PrimitiveType

    def __init__(
        self, indices: numpy.ndarray, primitive_type: PrimitiveType
    ) -> None: ...

class OutlineBuffer:
    attributes: list[AttributeData]

    def __init__(self, attributes: list[AttributeData]) -> None: ...

class PrimitiveType:
    TriangleList: ClassVar[PrimitiveType]
    QuadList: ClassVar[PrimitiveType]
    TriangleStrip: ClassVar[PrimitiveType]
    TriangleListAdjacency: ClassVar[PrimitiveType]

class UnkBuffer:
    unk2: int
    attributes: list[AttributeData]

    def __init__(self, unk2: int, attributes: list[AttributeData]) -> None: ...

class UnkDataBuffer:
    attribute1: numpy.ndarray
    attribute2: numpy.ndarray
    uniform_data: bytes
    unk: list[float]

    def __init__(
        self,
        attribute1: numpy.ndarray,
        attribute2: numpy.ndarray,
        uniform_data: bytes,
        unk: list[float],
    ) -> None: ...
