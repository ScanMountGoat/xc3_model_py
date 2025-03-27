from typing import Optional, ClassVar, Tuple
import numpy

from . import animation
from . import material
from . import monolib
from . import shader_database
from . import skinning
from . import vertex
from . import collision

def load_model(
    wimdo_path: str, shader_database: Optional[shader_database.ShaderDatabase]
) -> ModelRoot: ...
def load_model_legacy(
    camdo_path, shader_database: Optional[shader_database.ShaderDatabase]
) -> ModelRoot: ...
def load_map(
    wismhd: str, shader_database: Optional[shader_database.ShaderDatabase]
) -> list[MapRoot]: ...
def load_animations(anim_path: str) -> list[animation.Animation]: ...
def load_collisions(idcm_path: str) -> collision.CollisionMeshes: ...
def decode_images_png(
    images: list[ImageTexture], flip_vertical: bool
) -> list[bytes]: ...
def decode_images_rgbaf32(images: list[ImageTexture]) -> list[numpy.ndarray]: ...
def encode_images_rgba8(images: list[EncodeSurfaceRgba8Args]) -> list[ImageTexture]: ...
def encode_images_rgbaf32(
    images: list[EncodeSurfaceRgba32FloatArgs],
) -> list[ImageTexture]: ...

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
    materials: list[material.Material]
    samplers: list[Sampler]
    skinning: Optional[skinning.Skinning]
    lod_data: Optional[LodData]
    morph_controller_names: list[str]
    animation_morph_names: list[str]
    max_xyz: list[float]
    min_xyz: list[float]

    def __init__(
        self,
        models: list[Model],
        materials: list[material.Material],
        samplers: list[Sampler],
        max_xyz: list[float],
        min_xyz: list[float],
        morph_controller_names: list[str],
        animation_morph_names: list[str],
        skinning: Optional[skinning.Skinning],
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
    transform: Transform
    parent_index: Optional[int]

    def __init__(
        self, name: str, transform: Transform, parent_index: Optional[int]
    ) -> None: ...

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
    @staticmethod
    def from_dds(
        dds: Dds, name: Optional[str], usage: Optional[TextureUsage]
    ) -> ImageTexture: ...

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
    Unk21: ClassVar[TextureUsage]
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
    Unk16: ClassVar[TextureUsage]
    Unk22: ClassVar[TextureUsage]
    Unk23: ClassVar[TextureUsage]
    Unk24: ClassVar[TextureUsage]
    Unk25: ClassVar[TextureUsage]
    Unk26: ClassVar[TextureUsage]
    Unk27: ClassVar[TextureUsage]
    Unk28: ClassVar[TextureUsage]
    Unk29: ClassVar[TextureUsage]
    Unk30: ClassVar[TextureUsage]

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

class EncodeSurfaceRgba32FloatArgs:
    width: int
    height: int
    depth: int
    view_dimension: ViewDimension
    image_format: ImageFormat
    mipmaps: bool
    data: numpy.ndarray
    name: Optional[str]
    usage: Optional[TextureUsage]

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        view_dimension: ViewDimension,
        image_format: ImageFormat,
        mipmaps: bool,
        data: numpy.ndarray,
        name: Optional[str],
        usage: Optional[TextureUsage],
    ) -> None: ...
    def encode(self) -> ImageTexture: ...

class EncodeSurfaceRgba8Args:
    width: int
    height: int
    depth: int
    view_dimension: ViewDimension
    image_format: ImageFormat
    mipmaps: bool
    data: bytes
    name: Optional[str]
    usage: Optional[TextureUsage]

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        view_dimension: ViewDimension,
        image_format: ImageFormat,
        mipmaps: bool,
        data: bytes,
        name: Optional[str],
        usage: Optional[TextureUsage],
    ) -> None: ...
    def encode(self) -> ImageTexture: ...

class Dds:
    @staticmethod
    def from_file(path: str) -> Dds: ...
    def save(self, path: str): ...

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

class Mxmd:
    @staticmethod
    def from_file(path: str) -> Mxmd: ...
    def save(self, path: str): ...

class Msrd:
    @staticmethod
    def from_file(path: str) -> Msrd: ...
    def save(self, path: str): ...

class Transform:
    translation: list[float]
    rotation: list[float]
    scale: list[float]

    def __init__(
        self, translation: list[float], rotation: list[float], scale: list[float]
    ) -> None: ...
