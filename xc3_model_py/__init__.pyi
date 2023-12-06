from typing import List, Optional, ClassVar, Tuple, Dict
import numpy


def load_model(wimdo_path: str, database_path: Optional[str]) -> ModelRoot: ...


def load_map(wismhd: str, database_path: Optional[str]) -> List[ModelRoot]: ...


def load_animations(anim_path: str) -> List[Animation]: ...


def murmur3(name: str) -> int: ...


class ModelRoot:
    groups: List[ModelGroup]
    image_textures: List[ImageTexture]


class ModelGroup:
    models: List[Models]
    buffers: List[ModelBuffers]


class ModelBuffers:
    vertex_buffers: List[VertexBuffer]
    index_buffers: List[IndexBuffer]
    weights: Optional[Weights]


class Weights:
    skin_weights: SkinWeights


class SkinWeights:
    bone_indices: numpy.ndarray
    weights: numpy.ndarray
    bone_names: List[str]


class Models:
    models: List[Model]
    materials: List[Material]
    samplers: List[Sampler]
    skeleton: Optional[Skeleton]
    base_lod_indices: Optional[List[int]]


class Model:
    meshes: List[Mesh]
    instances: numpy.ndarray
    model_buffers_index: int


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
    def sampler_channel_index(self, output_index: int,
                              channel: str) -> Optional[Tuple[int, int]]: ...

    def float_constant(self, output_index: int,
                       channel: str) -> Optional[float]: ...

    def buffer_parameter(self, output_index: int,
                         channel: str) -> Optional[BufferParameter]: ...


class BufferParameter:
    buffer: str
    uniform: str
    index: int
    channel: str


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
    R4G4B4A4Unorm: ClassVar[ImageFormat]
    R16G16B16A16Float: ClassVar[ImageFormat]
    BC1Unorm: ClassVar[ImageFormat]
    BC2Unorm: ClassVar[ImageFormat]
    BC3Unorm: ClassVar[ImageFormat]
    BC4Unorm: ClassVar[ImageFormat]
    BC5Unorm: ClassVar[ImageFormat]
    BC56UFloat: ClassVar[ImageFormat]
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


class Animation:
    name: str
    space_mode: SpaceMode
    play_mode: PlayMode
    blend_mode: BlendMode
    frames_per_second: float
    frame_count: int
    tracks: List[Track]

    def current_frame(self, current_time_seconds: float) -> float: ...


class Track:
    def sample_translation(
        self, frame: float) -> Tuple[float, float, float]: ...

    def sample_rotation(self, frame: float) -> Tuple[float, float, float]: ...

    def sample_scale(self, frame: float) -> Tuple[float, float, float]: ...

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
