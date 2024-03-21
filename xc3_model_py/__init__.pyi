from typing import Optional, ClassVar, Tuple
import numpy


def load_model(wimdo_path: str, database_path: Optional[str]) -> ModelRoot: ...


def load_map(wismhd: str, database_path: Optional[str]) -> list[ModelRoot]: ...


def load_animations(anim_path: str) -> list[Animation]: ...


def murmur3(name: str) -> int: ...


class Xc3ModelError(Exception):
    pass


class ModelRoot:
    groups: list[ModelGroup]
    image_textures: list[ImageTexture]
    skeleton: Optional[Skeleton]


class ModelGroup:
    models: list[Models]
    buffers: list[ModelBuffers]


class ModelBuffers:
    vertex_buffers: list[VertexBuffer]
    index_buffers: list[IndexBuffer]
    weights: Optional[Weights]


class Weights:
    skin_weights: SkinWeights

    def weights_start_index(self, skin_flags: int,
                            lod: int, unk_type: RenderPassType) -> int: ...


class SkinWeights:
    bone_indices: numpy.ndarray
    weights: numpy.ndarray
    bone_names: list[str]


class Models:
    models: list[Model]
    materials: list[Material]
    samplers: list[Sampler]
    base_lod_indices: Optional[list[int]]


class Model:
    meshes: list[Mesh]
    instances: numpy.ndarray
    model_buffers_index: int


class Mesh:
    vertex_buffer_index: int
    index_buffer_index: int
    material_index: int
    lod: int
    skin_flags: int


class Skeleton:
    bones: list[Bone]

    def __init__(self, bones: list[Bone]) -> None: ...

    def model_space_transforms(self) -> numpy.ndarray: ...


class Bone:
    name: str
    transform: numpy.ndarray
    parent_index: Optional[int]

    def __init__(self, name: str, transform: numpy.ndarray,
                 parent_index: Optional[int]) -> None: ...


class Material:
    name: str
    textures: list[Texture]
    alpha_test: Optional[TextureAlphaTest]
    shader: Optional[Shader]
    pass_type: RenderPassType

    def output_assignments(
        self, textures: list[ImageTexture]) -> OutputAssignments: ...


class RenderPassType:
    Unk0: ClassVar[RenderPassType]
    Unk1: ClassVar[RenderPassType]
    Unk6: ClassVar[RenderPassType]
    Unk7: ClassVar[RenderPassType]
    Unk9: ClassVar[RenderPassType]


class TextureAlphaTest:
    texture_index: int
    channel_index: int
    ref_value: float


class MaterialParameters:
    mat_color: list[float]
    alpha_test_ref: float
    tex_matrix: Optional[list[float]]
    work_float4: Optional[list[float]]
    work_color: Optional[list[float]]


class Shader:
    pass


class Texture:
    image_texture_index: int
    sampler_index: int


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


class Influence:
    bone_name: str
    weights: list[SkinWeight]


class SkinWeight:
    vertex_index: int
    weight: float


class IndexBuffer:
    indices: numpy.ndarray


class ImageTexture:
    name: Optional[str]
    usage: Optional[TextureUsage]
    width: int
    height: int
    depth: int
    view_dimension: ViewDimension
    image_format: ImageFormat
    mipmap_count: int
    image_data: bytes


class TextureUsage:
    Unk0: ClassVar[TextureUsage]
    Temp: ClassVar[TextureUsage]
    Unk6: ClassVar[TextureUsage]
    Nrm: ClassVar[TextureUsage]
    Unk13: ClassVar[TextureUsage]
    WavePlus: ClassVar[TextureUsage]
    Col: ClassVar[TextureUsage]
    Unk8: ClassVar[TextureUsage]
    Alp: ClassVar[TextureUsage]
    Unk: ClassVar[TextureUsage]
    Alp2: ClassVar[TextureUsage]
    Col2: ClassVar[TextureUsage]
    Unk11: ClassVar[TextureUsage]
    Unk9: ClassVar[TextureUsage]
    Alp3: ClassVar[TextureUsage]
    Nrm2: ClassVar[TextureUsage]
    Col3: ClassVar[TextureUsage]
    Unk3: ClassVar[TextureUsage]
    Unk2: ClassVar[TextureUsage]
    Unk20: ClassVar[TextureUsage]
    Unk17: ClassVar[TextureUsage]
    F01: ClassVar[TextureUsage]
    Unk4: ClassVar[TextureUsage]
    Unk7: ClassVar[TextureUsage]
    Unk15: ClassVar[TextureUsage]
    Temp2: ClassVar[TextureUsage]
    Unk14: ClassVar[TextureUsage]
    Col4: ClassVar[TextureUsage]
    Alp4: ClassVar[TextureUsage]
    Unk12: ClassVar[TextureUsage]
    Unk18: ClassVar[TextureUsage]
    Unk19: ClassVar[TextureUsage]
    Unk5: ClassVar[TextureUsage]
    Unk10: ClassVar[TextureUsage]
    VolTex: ClassVar[TextureUsage]
    Unk1: ClassVar[TextureUsage]


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


class OutputAssignments:
    assignments: list[OutputAssignment]


class OutputAssignment:
    x: Optional[ChannelAssignment]
    y: Optional[ChannelAssignment]
    z: Optional[ChannelAssignment]
    w: Optional[ChannelAssignment]


class ChannelAssignment:
    def texture(self) -> Optional[ChannelAssignmentTexture]: ...

    def value(self) -> Optional[float]: ...


class ChannelAssignmentTexture:
    name: str
    channel_index: int
    texcoord_name: Optional[str]
    texcoord_scale: Optional[Tuple[float, float]]


class Animation:
    name: str
    space_mode: SpaceMode
    play_mode: PlayMode
    blend_mode: BlendMode
    frames_per_second: float
    frame_count: int
    tracks: list[Track]

    def current_frame(self, current_time_seconds: float) -> float: ...

    def skinning_transforms(self, skeleton: Skeleton,
                            frame: float) -> numpy.ndarray: ...

    def model_space_transforms(
        self, skeleton: Skeleton, frame: float) -> numpy.ndarray: ...


class Track:
    def sample_translation(
        self, frame: float) -> Optional[Tuple[float, float, float]]: ...

    def sample_rotation(
        self, frame: float) -> Optional[Tuple[float, float, float]]: ...

    def sample_scale(
        self, frame: float) -> Optional[Tuple[float, float, float]]: ...

    def sample_transform(self, frame: float) -> Optional[numpy.ndarray]: ...

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
