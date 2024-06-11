from typing import Optional, ClassVar, Tuple
import numpy

from . import animation
from . import skinning
from . import vertex

def load_model(wimdo_path: str, database_path: Optional[str]) -> ModelRoot: ...
def load_model_legacy(camdo_path) -> ModelRoot: ...
def load_map(wismhd: str, database_path: Optional[str]) -> list[MapRoot]: ...
def load_animations(anim_path: str) -> list[animation.Animation]: ...

class Xc3ModelError(Exception):
    pass

class ModelRoot:
    models: Models
    buffers: vertex.ModelBuffers
    image_textures: list[ImageTexture]
    skeleton: Optional[Skeleton]

    def __init__(
        self,
        models: Models,
        buffers: vertex.ModelBuffers,
        image_textures: list[ImageTexture],
        skeleton: Optional[Skeleton],
    ) -> None: ...
    def decode_images_rgbaf32(self) -> list[numpy.ndarray]: ...
    def save_images_rgba8(
        self, folder: str, prefix: str, ext: str, flip_vertical: bool
    ) -> list[str]: ...
    def to_mxmd_model(self, mxmd: Mxmd, msrd: Msrd) -> Tuple[Mxmd, Msrd]: ...

class MapRoot:
    groups: list[ModelGroup]
    image_textures: list[ImageTexture]

    def __init__(
        self, groups: list[ModelGroup], image_textures: list[ImageTexture]
    ) -> None: ...
    def decode_images_rgbaf32(self) -> list[numpy.ndarray]: ...
    def save_images_rgba8(
        self, folder: str, prefix: str, ext: str, flip_vertical: bool
    ) -> list[str]: ...

class ModelGroup:
    models: list[Models]
    buffers: list[vertex.ModelBuffers]

    def __init__(
        self, models: list[Models], buffers: list[vertex.ModelBuffers]
    ) -> None: ...

class Models:
    models: list[Model]
    materials: list[Material]
    samplers: list[Sampler]
    lod_data: Optional[LodData]
    morph_controller_names: list[str]
    animation_morph_names: list[str]
    max_xyz: list[float]
    min_xyz: list[float]

    def __init__(
        self,
        models: list[Model],
        materials: list[Material],
        samplers: list[Sampler],
        max_xyz: list[float],
        min_xyz: list[float],
        morph_controller_names: list[str],
        animation_morph_names: list[str],
        lod_data: Optional[LodData],
    ) -> None: ...

class Model:
    meshes: list[Mesh]
    instances: numpy.ndarray
    model_buffers_index: int
    max_xyz: list[float]
    min_xyz: list[float]
    bounding_radius: float

    def __init__(
        self,
        meshes: list[Mesh],
        instances: numpy.ndarray,
        model_buffers_index: int,
        max_xyz: list[float],
        min_xyz: list[float],
        bounding_radius: float,
    ) -> None: ...

class Mesh:
    vertex_buffer_index: int
    index_buffer_index: int
    index_buffer_index2: int
    material_index: int
    ext_mesh_index: Optional[int]
    lod_item_index: Optional[int]
    flags1: int
    flags2: int

    def __init__(
        self,
        vertex_buffer_index: int,
        index_buffer_index: int,
        index_buffer_index2: int,
        material_index: int,
        flags1: int,
        flags2: int,
        lod_item_index: Optional[int],
        ext_mesh_index: Optional[int],
        base_mesh_index: Optional[int],
    ) -> None: ...

class LodData:
    items: list[LodItem]
    groups: list[LodGroup]

    def __init__(self, items: list[LodItem], groups: list[LodGroup]) -> None: ...

class LodItem:
    unk2: float
    index: int
    unk5: int

    def __init__(self, unk2: float, index: int, unk5: int) -> None: ...

class LodGroup:
    base_lod_index: int
    lod_count: int

    def __init__(self, base_lod_index: int, lod_count: int) -> None: ...

class Skeleton:
    bones: list[Bone]

    def __init__(self, bones: list[Bone]) -> None: ...
    def model_space_transforms(self) -> numpy.ndarray: ...

class Bone:
    name: str
    transform: numpy.ndarray
    parent_index: Optional[int]

    def __init__(
        self, name: str, transform: numpy.ndarray, parent_index: Optional[int]
    ) -> None: ...

class Material:
    name: str
    textures: list[Texture]
    alpha_test: Optional[TextureAlphaTest]
    shader: Optional[Shader]
    pass_type: RenderPassType

    def __init__(
        self,
        name: str,
        textures: list[Texture],
        pass_type: RenderPassType,
        alpha_test: Optional[TextureAlphaTest],
        shader: Optional[Shader],
    ) -> None: ...
    def output_assignments(self, textures: list[ImageTexture]) -> OutputAssignments: ...

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

    def __init__(
        self, texture_index: int, channel_index: int, ref_value: float
    ) -> None: ...

class MaterialParameters:
    mat_color: list[float]
    alpha_test_ref: float
    tex_matrix: Optional[list[float]]
    work_float4: Optional[list[float]]
    work_color: Optional[list[float]]

    def __init__(
        self,
        mat_color: list[float],
        alpha_test_ref: float,
        tex_matrix: Optional[list[float]],
        work_float4: Optional[list[float]],
        work_color: Optional[list[float]],
    ) -> None: ...

class Shader:
    pass

class Texture:
    image_texture_index: int
    sampler_index: int

    def __init__(self, image_texture_index: int, sampler_index: int) -> None: ...

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

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        view_dimension: ViewDimension,
        image_format: ImageFormat,
        mipmap_count: int,
        image_data: bytes,
        name: Optional[str],
        usage: Optional[TextureUsage],
    ) -> None: ...

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

    def __init__(
        self,
        address_mode_u: AddressMode,
        address_mode_v: AddressMode,
        address_mode_w: AddressMode,
        min_filter: FilterMode,
        mag_filter: FilterMode,
        mip_filter: FilterMode,
        mipmaps: bool,
    ) -> None: ...

class AddressMode:
    ClampToEdge: ClassVar[AddressMode]
    Repeat: ClassVar[AddressMode]
    MirrorRepeat: ClassVar[AddressMode]

class FilterMode:
    Nearest: ClassVar[FilterMode]
    Linear: ClassVar[FilterMode]

class OutputAssignments:
    assignments: list[OutputAssignment]

    def mat_id(self) -> Optional[int]: ...

class OutputAssignment:
    x: Optional[ChannelAssignment]
    y: Optional[ChannelAssignment]
    z: Optional[ChannelAssignment]
    w: Optional[ChannelAssignment]

class ChannelAssignment:
    def texture(self) -> Optional[ChannelAssignmentTexture]: ...
    def value(self) -> Optional[float]: ...
    def attribute(self) -> Optional[ChannelAssignmentAttribute]: ...

class ChannelAssignmentTexture:
    name: str
    channel_index: int
    texcoord_name: Optional[str]
    texcoord_scale: Optional[Tuple[float, float]]

class ChannelAssignmentAttribute:
    name: str
    channel_index: int

class Mxmd:
    @staticmethod
    def from_file(path: str) -> Mxmd: ...
    def save(self, path: str): ...

class Msrd:
    @staticmethod
    def from_file(path: str) -> Msrd: ...
    def save(self, path: str): ...
