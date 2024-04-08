from typing import Optional, ClassVar, Tuple
import numpy

from xc3_model_py import Weights
from xc3_model_py.skinning import Influence

class ModelBuffers:
    vertex_buffers: list[VertexBuffer]
    index_buffers: list[IndexBuffer]
    weights: Optional[Weights]


class VertexBuffer:
    attributes: list[AttributeData]
    influences: list[Influence]


class AttributeData:
    attribute_type: AttributeType
    data: numpy.ndarray


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
    SkinWeights: ClassVar[AttributeType]
    BoneIndices: ClassVar[AttributeType]


class MorphTarget:
    position_deltas: numpy.ndarray
    normal_deltas: numpy.ndarray
    tangent_deltas: numpy.ndarray


class IndexBuffer:
    indices: numpy.ndarray
