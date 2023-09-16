from typing import List, Optional, ClassVar, Tuple
import numpy


def load_model(wimdo_path: str) -> ModelRoot: ...


def load_map(wismhd: str) -> List[ModelRoot]: ...


class ModelRoot:
    groups: List[ModelGroup]
    image_textures: List[ImageTexture]


class ModelGroup:
    models: List[Model]
    materials: List[Material]
    skeleton: Optional[Skeleton]


class Model:
    meshes: List[Mesh]
    vertex_buffers: List[VertexBuffer]
    index_buffers: List[IndexBuffer]
    instances: numpy.ndarray


class Mesh:
    vertex_buffer_index: int
    index_buffer_index: int
    material_index: int


class Skeleton:
    bones: List[Bone]


class Bone:
    name: str
    transform: numpy.ndarray
    parent_index: Optional[int]


class Material:
    name: str
    textures: List[Texture]
    shader: Optional[Shader]


class TextureAlphaTest:
    texture_index: int
    channel_index: int
    ref_value: float


class MaterialParameters:
    mat_color: Tuple[float, float, float, float]
    alpha_test_ref: float
    tex_matrix: Optional[List[float]]
    work_float4: Optional[Tuple[float, float, float, float]]
    work_color: Optional[Tuple[float, float, float, float]]


class Shader:
    output_dependencies: dict[str, List[str]]


class Texture:
    image_texture_index: int


class VertexBuffer:
    attributes: List[AttributeData]
    influences: List[Influence]


class AttributeData:
    attribute_type: AttributeType
    data: numpy.ndarray


class AttributeType:
    Position: ClassVar[AttributeType]
    Normal: ClassVar[AttributeType]
    Tangent: ClassVar[AttributeType]
    Uv1: ClassVar[AttributeType]
    Uv2: ClassVar[AttributeType]
    VertexColor: ClassVar[AttributeType]
    WeightIndex: ClassVar[AttributeType]
    SkinWeights: ClassVar[AttributeType]
    BoneIndices: ClassVar[AttributeType]


class Influence:
    bone_name: str
    weights: List[SkinWeight]


class SkinWeight:
    vertex_index: int
    weight: float


class IndexBuffer:
    indices: numpy.ndarray


class ImageTexture:
    name: Optional[str]
    width: int
    height: int
    depth: int
    view_dimension: ViewDimension
    image_format: ImageFormat
    mipmap_count: int
    image_data: bytes


class ViewDimension:
    D2: ClassVar[ViewDimension]
    D3: ClassVar[ViewDimension]
    Cube: ClassVar[ViewDimension]


class ImageFormat:
    R8Unorm: ClassVar[ImageFormat]
    R8G8B8A8Unorm: ClassVar[ImageFormat]
    R16G16B16A16Float: ClassVar[ImageFormat]
    BC1Unorm: ClassVar[ImageFormat]
    BC2Unorm: ClassVar[ImageFormat]
    BC3Unorm: ClassVar[ImageFormat]
    BC4Unorm: ClassVar[ImageFormat]
    BC5Unorm: ClassVar[ImageFormat]
    BC7Unorm: ClassVar[ImageFormat]
    B8G8R8A8Unorm: ClassVar[ImageFormat]


class Sampler:
    address_mode_u: AddressMode
    address_mode_v: AddressMode
    address_mode_w: AddressMode
    min_filter: FilterMode
    mag_filter: FilterMode
    mipmaps: bool


class AddressMode:
    ClampToEdge: ClassVar[AddressMode]
    Repeat: ClassVar[AddressMode]
    MirrorRepeat: ClassVar[AddressMode]


class FilterMode:
    Nearest: ClassVar[FilterMode]
    Linear: ClassVar[FilterMode]
