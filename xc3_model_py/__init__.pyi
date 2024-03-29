from typing import Optional, ClassVar, Tuple
import numpy

from . import animation
from . import skinning
from . import vertex


def load_model(wimdo_path: str, database_path: Optional[str]) -> ModelRoot: ...


def load_map(wismhd: str, database_path: Optional[str]) -> list[ModelRoot]: ...


def load_animations(anim_path: str) -> list[animation.Animation]: ...


class Xc3ModelError(Exception):
    pass


class ModelRoot:
    groups: list[ModelGroup]
    image_textures: list[ImageTexture]
    skeleton: Optional[Skeleton]

    def to_mxmd_model(self, mxmd: Mxmd, msrd: Msrd) -> Tuple[Mxmd, Msrd]: ...


class ModelGroup:
    models: list[Models]
    buffers: list[vertex.ModelBuffers]


class Weights:
    skin_weights: skinning.SkinWeights

    def weights_start_index(self, skin_flags: int,
                            lod: int, unk_type: RenderPassType) -> int: ...


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
    mip_filter: FilterMode
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


class Mxmd:
    @staticmethod
    def from_file(path: str) -> Mxmd: ...

    def save(self, path: str): ...


class Msrd:
    @staticmethod
    def from_file(path: str) -> Msrd: ...

    def save(self, path: str): ...
