from typing import Optional, ClassVar, Tuple

from xc3_model_py.xc3_model_py import ImageTexture
from xc3_model_py.shader_database import Operation, ShaderProgram

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
    work_callbacks: list[WorkCallback]
    alpha_test_ref: float
    m_unks1_1: int
    m_unks1_2: int
    m_unks1_3: int
    m_unks1_4: int
    shader: Optional[ShaderProgram]
    technique_index: int
    pass_type: RenderPassType
    parameters: MaterialParameters
    m_unks2_2: int
    gbuffer_flags: int
    fur_params: Optional[FurShellParams]

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
        alpha_test_ref: float,
        m_unks1_1: int,
        m_unks1_2: int,
        m_unks1_3: int,
        m_unks1_4: int,
        technique_index: int,
        pass_type: RenderPassType,
        parameters: MaterialParameters,
        m_unks2_2: int,
        gbuffer_flags: int,
        alpha_test: Optional[TextureAlphaTest],
        shader: Optional[ShaderProgram],
        fur_params: Optional[FurShellParams],
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
    Unk128: ClassVar[StencilValue]

class StencilMode:
    Unk0: ClassVar[StencilMode]
    Unk1: ClassVar[StencilMode]
    Unk2: ClassVar[StencilMode]
    Unk6: ClassVar[StencilMode]
    Unk7: ClassVar[StencilMode]
    Unk8: ClassVar[StencilMode]
    Unk9: ClassVar[StencilMode]
    Unk12: ClassVar[StencilMode]
    Unk13: ClassVar[StencilMode]

class ColorWriteMode:
    Unk0: ClassVar[ColorWriteMode]
    Unk1: ClassVar[ColorWriteMode]
    Unk2: ClassVar[ColorWriteMode]
    Unk3: ClassVar[ColorWriteMode]
    Unk5: ClassVar[ColorWriteMode]
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
    sampler_index: int
    channel_index: int

    def __init__(
        self, texture_index: int, sampler_index: int, channel_index: int
    ) -> None: ...

class MaterialParameters:
    material_color: list[float]
    tex_matrix: Optional[list[float]]
    work_float4: Optional[list[float]]
    work_color: Optional[list[float]]
    alpha_info: Optional[list[float]]
    dp_rat: Optional[list[float]]
    projection_tex_matrix: Optional[list[float]]
    material_ambient: Optional[list[float]]
    material_specular: Optional[list[float]]
    dt_work: Optional[list[float]]
    mdl_param: Optional[list[float]]
    ava_skin: Optional[list[float]]

    def __init__(
        self,
        material_color: list[float],
        tex_matrix: Optional[list[float]],
        work_float4: Optional[list[float]],
        work_color: Optional[list[float]],
        ava_skin: Optional[list[float]],
    ) -> None: ...

class Texture:
    image_texture_index: int
    sampler_index: int

    def __init__(self, image_texture_index: int, sampler_index: int) -> None: ...

class WorkCallback:
    unk1: int
    unk2: int

    def __init__(self, unk1: int, unk2: int) -> None: ...

class FurShellParams:
    instance_count: int
    view_distance: float
    shell_width: float
    y_offset: float
    alpha: float

    def __init__(
        self,
        instance_count: int,
        view_distance: float,
        shell_width: float,
        y_offset: float,
        alpha: float,
    ) -> None: ...

class OutputAssignments:
    output_assignments: list[OutputAssignment]
    outline_width: Optional[AssignmentValue]
    normal_intensity: Optional[int]
    assignments: list[Assignment]

    def mat_id(self) -> Optional[int]: ...

class OutputAssignment:
    x: Optional[int]
    y: Optional[int]
    z: Optional[int]
    w: Optional[int]

    def merge_xyz(
        self, assignments: list[Assignment]
    ) -> Optional[OutputAssignmentXyz]: ...

class Assignment:
    def func(self) -> Optional[AssignmentFunc]: ...
    def value(self) -> Optional[AssignmentValue]: ...

class AssignmentValue:
    def texture(self) -> Optional[TextureAssignment]: ...
    def float(self) -> Optional[float]: ...
    def attribute(self) -> Optional[AssignmentValueAttribute]: ...

class AssignmentFunc:
    op: Operation
    args: list[int]

class TextureAssignment:
    name: str
    channel: Optional[str]
    texcoords: list[int]

class AssignmentValueAttribute:
    name: str
    channel: Optional[str]

class OutputAssignmentXyz:
    assignment: int
    assignments: list[AssignmentXyz]

class AssignmentXyz:
    def func(self) -> Optional[AssignmentFuncXyz]: ...
    def value(self) -> Optional[AssignmentValueXyz]: ...

class AssignmentValueXyz:
    def texture(self) -> Optional[TextureAssignmentXyz]: ...
    def float(self) -> Optional[Tuple[float, float, float]]: ...
    def attribute(self) -> Optional[AssignmentValueAttributeXyz]: ...

class AssignmentFuncXyz:
    op: Operation
    args: list[int]

class TextureAssignmentXyz:
    name: str
    channel: Optional[ChannelXyz]
    texcoords: list[int]

class AssignmentValueAttributeXyz:
    name: str
    channel: Optional[ChannelXyz]

class ChannelXyz:
    Xyz: ClassVar[ChannelXyz]
    X: ClassVar[ChannelXyz]
    Y: ClassVar[ChannelXyz]
    Z: ClassVar[ChannelXyz]
    W: ClassVar[ChannelXyz]
