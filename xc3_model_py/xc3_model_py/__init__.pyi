from typing import Optional, ClassVar, Tuple
import numpy

from . import animation
from . import skinning
from . import vertex
from . import shader_database

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
    flags: int
    render_flags: int
    state_flags: StateFlags
    color: list[float]
    textures: list[Texture]
    alpha_test: Optional[TextureAlphaTest]
    work_values: list[float]
    shader_vars: list[Tuple[int, int]]
    work_callbacks: list[Tuple[int, int]]
    alpha_test_ref: list[int]
    m_unks1_1: int
    m_unks1_2: int
    m_unks1_3: int
    m_unks1_4: int
    shader: Optional[shader_database.ShaderProgram]
    technique_index: int
    pass_type: RenderPassType
    parameters: MaterialParameters
    m_unks2_2: int
    m_unks3_1: int

    def __init__(
        self,
        name: str,
        flags: int,
        render_flags: int,
        state_flags: StateFlags,
        color: list[float],
        textures: list[Texture],
        work_values: list[float],
        shader_vars: list[Tuple[int, int]],
        work_callbacks: list[Tuple[int, int]],
        alpha_test_ref: list[int],
        m_unks1_1: int,
        m_unks1_2: int,
        m_unks1_3: int,
        m_unks1_4: int,
        technique_index: int,
        pass_type: RenderPassType,
        parameters: MaterialParameters,
        m_unks2_2: int,
        m_unks3_1: int,
        alpha_test: Optional[TextureAlphaTest],
        shader: Optional[shader_database.ShaderProgram],
    ) -> None: ...
    def output_assignments(self, textures: list[ImageTexture]) -> OutputAssignments: ...

class RenderPassType:
    Unk0: ClassVar[RenderPassType]
    Unk1: ClassVar[RenderPassType]
    Unk6: ClassVar[RenderPassType]
    Unk7: ClassVar[RenderPassType]
    Unk9: ClassVar[RenderPassType]

class StateFlags:
    depth_write_mode: int
    blend_mode: BlendMode
    cull_mode: CullMode
    unk4: int
    stencil_value: StencilValue
    stencil_mode: StencilMode
    depth_func: DepthFunc
    color_write_mode: int

class BlendMode:
    Disabled: ClassVar[BlendMode]
    Blend: ClassVar[BlendMode]
    Unk2: ClassVar[BlendMode]
    Multiply: ClassVar[BlendMode]
    MultiplyInverted: ClassVar[BlendMode]
    Add: ClassVar[BlendMode]
    Disabled2: ClassVar[BlendMode]

class CullMode:
    Back: ClassVar[CullMode]
    Front: ClassVar[CullMode]
    Disabled: ClassVar[CullMode]
    Unk3: ClassVar[CullMode]

class StencilValue:
    Unk0: ClassVar[StencilValue]
    Unk1: ClassVar[StencilValue]
    Unk4: ClassVar[StencilValue]
    Unk5: ClassVar[StencilValue]
    Unk8: ClassVar[StencilValue]
    Unk9: ClassVar[StencilValue]
    Unk12: ClassVar[StencilValue]
    Unk16: ClassVar[StencilValue]
    Unk20: ClassVar[StencilValue]
    Unk33: ClassVar[StencilValue]
    Unk37: ClassVar[StencilValue]
    Unk41: ClassVar[StencilValue]
    Unk49: ClassVar[StencilValue]
    Unk97: ClassVar[StencilValue]
    Unk105: ClassVar[StencilValue]

class StencilMode:
    Unk0: ClassVar[StencilMode]
    Unk1: ClassVar[StencilMode]
    Unk2: ClassVar[StencilMode]
    Unk6: ClassVar[StencilMode]
    Unk7: ClassVar[StencilMode]
    Unk8: ClassVar[StencilMode]

class ColorWriteMode:
    Unk0: ClassVar[ColorWriteMode]
    Unk1: ClassVar[ColorWriteMode]
    Unk2: ClassVar[ColorWriteMode]
    Unk3: ClassVar[ColorWriteMode]
    Unk6: ClassVar[ColorWriteMode]
    Unk9: ClassVar[ColorWriteMode]
    Unk10: ClassVar[ColorWriteMode]
    Unk11: ClassVar[ColorWriteMode]
    Unk12: ClassVar[ColorWriteMode]

class DepthFunc:
    Disabled: ClassVar[DepthFunc]
    LessEqual: ClassVar[DepthFunc]
    Equal: ClassVar[DepthFunc]

class TextureAlphaTest:
    texture_index: int
    channel_index: int
    ref_value: float

    def __init__(
        self, texture_index: int, channel_index: int, ref_value: float
    ) -> None: ...

class MaterialParameters:
    alpha_test_ref: float
    tex_matrix: Optional[list[float]]
    work_float4: Optional[list[float]]
    work_color: Optional[list[float]]

    def __init__(
        self,
        alpha_test_ref: float,
        tex_matrix: Optional[list[float]],
        work_float4: Optional[list[float]],
        work_color: Optional[list[float]],
    ) -> None: ...

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
    def textures(self) -> Optional[list[TextureAssignment]]: ...
    def value(self) -> Optional[float]: ...
    def attribute(self) -> Optional[ChannelAssignmentAttribute]: ...

class TextureAssignment:
    name: str
    channels: str
    texcoord_name: Optional[str]
    texcoord_transforms: Optional[
        Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]
    ]

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
