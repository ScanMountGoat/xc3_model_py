use pyo3::prelude::*;

use crate::python_enum;

python_enum!(
    BlendMode,
    xc3_model::material::BlendMode,
    Disabled,
    Blend,
    Unk2,
    Multiply,
    MultiplyInverted,
    Add,
    Disabled2
);

python_enum!(
    CullMode,
    xc3_model::material::CullMode,
    Back,
    Front,
    Disabled,
    Unk3
);

python_enum!(
    StencilValue,
    xc3_model::material::StencilValue,
    Unk0,
    Unk1,
    Unk4,
    Unk5,
    Unk8,
    Unk9,
    Unk12,
    Unk16,
    Unk20,
    Unk33,
    Unk37,
    Unk39,
    Unk41,
    Unk49,
    Unk65,
    Unk97,
    Unk105,
    Unk128
);

python_enum!(
    StencilMode,
    xc3_model::material::StencilMode,
    Unk0,
    Unk1,
    Unk2,
    Unk4,
    Unk6,
    Unk7,
    Unk8,
    Unk9,
    Unk12,
    Unk13
);

python_enum!(
    DepthFunc,
    xc3_model::material::DepthFunc,
    Disabled,
    LessEqual,
    Equal
);

python_enum!(
    ColorWriteMode,
    xc3_model::material::ColorWriteMode,
    Unk0,
    Unk1,
    Unk2,
    Unk3,
    Unk5,
    Unk6,
    Unk9,
    Unk10,
    Unk11,
    Unk12
);

python_enum!(
    TextureUsage,
    xc3_model::material::TextureUsage,
    Unk0,
    Temp,
    Unk6,
    Nrm,
    Unk13,
    WavePlus,
    Col,
    Unk8,
    Alp,
    Unk,
    Unk21,
    Alp2,
    Col2,
    Unk11,
    Unk9,
    Alp3,
    Nrm2,
    Col3,
    Unk3,
    Unk2,
    Unk20,
    Unk17,
    F01,
    Unk4,
    Unk7,
    Unk15,
    Temp2,
    Unk14,
    Col4,
    Alp4,
    Unk12,
    Unk18,
    Unk19,
    Unk5,
    Unk10,
    VolTex,
    Unk16,
    Unk22,
    Unk23,
    Unk24,
    Unk25,
    Unk26,
    Unk27,
    Unk28,
    Unk29,
    Unk30,
    Unk31
);

python_enum!(
    RenderPassType,
    xc3_model::material::RenderPassType,
    Unk0,
    Unk1,
    Unk2,
    Unk3,
    Unk5,
    Unk6,
    Unk7,
    Unk8,
    Unk9
);

python_enum!(
    ChannelXyz,
    xc3_model::material::assignments::ChannelXyz,
    Xyz,
    X,
    Y,
    Z,
    W
);

#[pymodule]
pub mod material {
    use crate::shader_database::Operation;
    use crate::shader_database::shader_database::ShaderProgram;
    use crate::xc3_model_py::ImageTexture;
    use map_py::{MapPy, TypedList};
    use numpy::PyArray1;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    #[pymodule_export]
    use super::BlendMode;

    #[pymodule_export]
    use super::CullMode;

    #[pymodule_export]
    use super::StencilValue;

    #[pymodule_export]
    use super::StencilMode;

    #[pymodule_export]
    use super::DepthFunc;

    #[pymodule_export]
    use super::ColorWriteMode;

    #[pymodule_export]
    use super::TextureUsage;

    #[pymodule_export]
    use super::RenderPassType;

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::Material)]
    pub struct Material {
        pub name: String,
        pub flags: MaterialFlags,
        // TODO: how to handle flags?
        #[map(from(map_py::helpers::into), into(map_py::helpers::into))]
        pub render_flags: u32,
        pub state_flags: StateFlags,
        pub color: [f32; 4],
        pub textures: TypedList<Texture>,
        pub alt_textures: Option<TypedList<Texture>>,

        #[map(from(map_py::helpers::into_option_py))]
        #[map(into(map_py::helpers::from_option_py))]
        pub alpha_test: Option<Py<Texture>>,

        pub work_values: Py<PyArray1<f32>>,
        pub shader_vars: Vec<(u16, u16)>,
        pub work_callbacks: TypedList<WorkCallback>,
        pub alpha_test_ref: f32,
        pub m_unks1_1: u32,
        pub m_unks1_2: u32,
        pub m_unks1_3: u32,
        pub m_unks1_4: u32,

        #[map(from(map_py::helpers::into_option_py))]
        #[map(into(map_py::helpers::from_option_py))]
        pub shader: Option<Py<ShaderProgram>>,

        pub technique_index: usize,
        pub pass_type: RenderPassType,

        #[map(from(map_py::helpers::into_py), into(map_py::helpers::from_py))]
        pub parameters: Py<MaterialParameters>,

        pub m_unks2: u16,
        pub gbuffer_flags: u16,

        #[map(from(map_py::helpers::into_option_py))]
        #[map(into(map_py::helpers::from_option_py))]
        pub fur_params: Option<Py<FurShellParams>>,
    }

    #[pymethods]
    impl Material {
        #[new]
        fn new(
            name: String,
            flags: MaterialFlags,
            render_flags: u32,
            state_flags: StateFlags,
            color: [f32; 4],
            textures: TypedList<Texture>,
            alt_textures: Option<TypedList<Texture>>,
            work_values: Py<PyArray1<f32>>,
            shader_vars: Vec<(u16, u16)>,
            work_callbacks: TypedList<WorkCallback>,
            alpha_test_ref: f32,
            m_unks1_1: u32,
            m_unks1_2: u32,
            m_unks1_3: u32,
            m_unks1_4: u32,
            technique_index: usize,
            pass_type: RenderPassType,
            parameters: Py<MaterialParameters>,
            m_unks2: u16,
            gbuffer_flags: u16,
            alpha_test: Option<Py<Texture>>,
            shader: Option<Py<ShaderProgram>>,
            fur_params: Option<Py<FurShellParams>>,
        ) -> Self {
            Self {
                name,
                flags,
                render_flags,
                state_flags,
                color,
                textures,
                alt_textures,
                alpha_test,
                work_values,
                shader_vars,
                work_callbacks,
                alpha_test_ref,
                m_unks1_1,
                m_unks1_2,
                m_unks1_3,
                m_unks1_4,
                shader,
                technique_index,
                pass_type,
                parameters,
                m_unks2,
                gbuffer_flags,
                fur_params,
            }
        }

        pub fn output_assignments(
            &self,
            py: Python,
            textures: Vec<PyRef<ImageTexture>>,
        ) -> PyResult<OutputAssignments> {
            // Converting all the Python images to Rust is very expensive.
            // We can avoid costly conversion of input images using PyRef.
            // We only need certain fields, so we can cheat a little here.
            let image_textures: Vec<_> = textures
                .iter()
                .map(|t| xc3_model::ImageTexture {
                    name: t.name.clone(),
                    usage: t.usage.map(Into::into),
                    width: 1,
                    height: 1,
                    depth: 1,
                    view_dimension: xc3_model::ViewDimension::D2,
                    image_format: xc3_model::ImageFormat::BC7Unorm,
                    mipmap_count: 1,
                    image_data: Vec::new(),
                })
                .collect();

            let assignments: xc3_model::material::assignments::OutputAssignments =
                self.clone().map_py(py)?.output_assignments(&image_textures);
            assignments.map_py(py)
        }

        pub fn alpha_texture_channel_index(&self, py: Python) -> PyResult<usize> {
            Ok(self.clone().map_py(py)?.alpha_texture_channel_index())
        }

        fn __deepcopy__(&self, py: Python, _memo: Py<PyDict>) -> Self {
            let copy: xc3_model::material::Material = self.clone().map_py(py).unwrap();
            copy.map_py(py).unwrap()
        }
    }

    // TODO: macro for generating python class for bilge bitfield.
    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct MaterialFlags {
        pub unk1: bool,
        pub unk2: bool,
        pub alpha_mask: bool,
        pub separate_mask: bool,
        pub unk5: bool,
        pub unk6: bool,
        pub unk7: bool,
        pub unk8: bool,
        pub unk9: bool,
        pub fur: bool,
        pub unk11: u32,
        pub fur_shells: bool,
        pub unk: u8,
    }

    #[pymethods]
    impl MaterialFlags {
        #[new]
        fn new(
            unk1: bool,
            unk2: bool,
            alpha_mask: bool,
            separate_mask: bool,
            unk5: bool,
            unk6: bool,
            unk7: bool,
            unk8: bool,
            unk9: bool,
            fur: bool,
            unk11: u32,
            fur_shells: bool,
            unk: u8,
        ) -> Self {
            Self {
                unk1,
                unk2,
                alpha_mask,
                separate_mask,
                unk5,
                unk6,
                unk7,
                unk8,
                unk9,
                fur,
                unk11,
                fur_shells,
                unk,
            }
        }
    }

    impl MapPy<xc3_lib::mxmd::MaterialFlags> for MaterialFlags {
        fn map_py(self, _py: Python) -> PyResult<xc3_lib::mxmd::MaterialFlags> {
            Ok(xc3_lib::mxmd::MaterialFlags::new(
                self.unk1,
                self.unk2,
                self.alpha_mask,
                self.separate_mask,
                self.unk5,
                self.unk6,
                self.unk7,
                self.unk8,
                self.unk9,
                self.fur,
                bilge::arbitrary_int::u17::new(self.unk11),
                self.fur_shells,
                bilge::arbitrary_int::u4::new(self.unk),
            ))
        }
    }

    impl MapPy<MaterialFlags> for xc3_lib::mxmd::MaterialFlags {
        fn map_py(self, _py: Python) -> PyResult<MaterialFlags> {
            Ok(MaterialFlags {
                unk1: self.unk1(),
                unk2: self.unk2(),
                alpha_mask: self.alpha_mask(),
                separate_mask: self.separate_mask(),
                unk5: self.unk5(),
                unk6: self.unk6(),
                unk7: self.unk7(),
                unk8: self.unk8(),
                unk9: self.unk9(),
                fur: self.fur(),
                unk11: self.unk11().value(),
                fur_shells: self.fur_shells(),
                unk: self.unk().value(),
            })
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::StateFlags)]
    pub struct StateFlags {
        pub depth_write_mode: u8,
        pub blend_mode: BlendMode,
        pub cull_mode: CullMode,
        pub unk4: u8,
        pub stencil_value: StencilValue,
        pub stencil_mode: StencilMode,
        pub depth_func: DepthFunc,
        pub color_write_mode: ColorWriteMode,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::MaterialParameters)]
    pub struct MaterialParameters {
        pub material_color: [f32; 4],
        pub tex_matrix: Option<Vec<[f32; 4]>>,
        pub work_float4: Option<Vec<[f32; 4]>>,
        pub work_color: Option<Vec<[f32; 4]>>,
        pub alpha_info: Option<Vec<[f32; 4]>>,
        pub dp_rat: Option<Vec<[f32; 4]>>,
        pub projection_tex_matrix: Option<Vec<[f32; 4]>>,
        pub material_ambient: Option<Vec<[f32; 4]>>,
        pub material_specular: Option<Vec<[f32; 4]>>,
        pub dt_work: Option<Vec<[f32; 4]>>,
        pub mdl_param: Option<Vec<[f32; 4]>>,
        pub ava_skin: Option<[f32; 4]>,
    }

    #[pymethods]
    impl MaterialParameters {
        #[new]
        fn new(
            material_color: [f32; 4],
            tex_matrix: Option<Vec<[f32; 4]>>,
            work_float4: Option<Vec<[f32; 4]>>,
            work_color: Option<Vec<[f32; 4]>>,
            alpha_info: Option<Vec<[f32; 4]>>,
            dp_rat: Option<Vec<[f32; 4]>>,
            projection_tex_matrix: Option<Vec<[f32; 4]>>,
            material_ambient: Option<Vec<[f32; 4]>>,
            material_specular: Option<Vec<[f32; 4]>>,
            dt_work: Option<Vec<[f32; 4]>>,
            mdl_param: Option<Vec<[f32; 4]>>,
            ava_skin: Option<[f32; 4]>,
        ) -> Self {
            Self {
                material_color,
                tex_matrix,
                work_float4,
                work_color,
                alpha_info,
                dp_rat,
                projection_tex_matrix,
                material_ambient,
                material_specular,
                dt_work,
                mdl_param,
                ava_skin,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::Texture)]
    pub struct Texture {
        pub image_texture_index: usize,
        pub sampler_index: usize,
        pub sampler_index2: usize,
    }

    #[pymethods]
    impl Texture {
        #[new]
        fn new(image_texture_index: usize, sampler_index: usize, sampler_index2: usize) -> Self {
            Self {
                image_texture_index,
                sampler_index,
                sampler_index2,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::WorkCallback)]
    pub struct WorkCallback {
        pub unk1: u16,
        pub unk2: u16,
    }

    #[pymethods]
    impl WorkCallback {
        #[new]
        fn new(unk1: u16, unk2: u16) -> Self {
            Self { unk1, unk2 }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::FurShellParams)]
    pub struct FurShellParams {
        pub instance_count: u32,
        pub view_distance: f32,
        pub shell_width: f32,
        pub y_offset: f32,
        pub alpha: f32,
    }

    #[pymethods]
    impl FurShellParams {
        #[new]
        fn new(
            instance_count: u32,
            view_distance: f32,
            shell_width: f32,
            y_offset: f32,
            alpha: f32,
        ) -> Self {
            Self {
                instance_count,
                view_distance,
                shell_width,
                y_offset,
                alpha,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::assignments::OutputAssignments)]
    pub struct OutputAssignments {
        pub output_assignments: [OutputAssignment; 6],
        pub outline_width: Option<AssignmentValue>,
        pub normal_intensity: Option<usize>,
        pub assignments: TypedList<Assignment>,
    }

    #[pymethods]
    impl OutputAssignments {
        fn mat_id(&self, py: Python) -> PyResult<Option<u32>> {
            let assignments: xc3_model::material::assignments::OutputAssignments =
                self.clone().map_py(py)?;
            Ok(assignments.mat_id())
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::assignments::OutputAssignment)]
    pub struct OutputAssignment {
        pub x: Option<usize>,
        pub y: Option<usize>,
        pub z: Option<usize>,
        pub w: Option<usize>,
    }

    #[pymethods]
    impl OutputAssignment {
        fn merge_xyz(
            &self,
            py: Python,
            assignments: TypedList<Assignment>,
        ) -> PyResult<Option<OutputAssignmentXyz>> {
            let output_assignment: xc3_model::material::assignments::OutputAssignment =
                self.clone().map_py(py)?;
            let assignments: Vec<xc3_model::material::assignments::Assignment> =
                assignments.map_py(py)?;
            output_assignment
                .merge_xyz(&assignments)
                .map(|o| o.map_py(py))
                .transpose()
        }
    }

    #[pyclass]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::assignments::Assignment)]
    pub struct Assignment(pub xc3_model::material::assignments::Assignment);

    #[pyclass]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::assignments::AssignmentValue)]
    pub struct AssignmentValue(pub xc3_model::material::assignments::AssignmentValue);

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct AssignmentFunc {
        pub op: Operation,
        pub args: Vec<usize>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct TextureAssignment {
        pub name: String,
        pub channel: Option<char>,
        pub texcoords: Vec<usize>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct AssignmentValueAttribute {
        pub name: String,
        pub channel: Option<char>,
    }

    #[pymethods]
    impl Assignment {
        pub fn func(&self) -> Option<AssignmentFunc> {
            match &self.0 {
                xc3_model::material::assignments::Assignment::Func { op, args } => {
                    Some(AssignmentFunc {
                        op: (*op).into(),
                        args: args.clone(),
                    })
                }
                _ => None,
            }
        }

        pub fn value(&self) -> Option<AssignmentValue> {
            match &self.0 {
                xc3_model::material::assignments::Assignment::Value(v) => {
                    Some(AssignmentValue(v.clone()?))
                }
                _ => None,
            }
        }
    }

    #[pymethods]
    impl AssignmentValue {
        pub fn texture(&self) -> Option<TextureAssignment> {
            match &self.0 {
                xc3_model::material::assignments::AssignmentValue::Texture(t) => {
                    Some(TextureAssignment {
                        name: t.name.to_string(),
                        channel: t.channel,
                        texcoords: t.texcoords.clone(),
                    })
                }
                _ => None,
            }
        }

        pub fn float(&self) -> Option<f32> {
            match self.0 {
                xc3_model::material::assignments::AssignmentValue::Float(f) => Some(f.0),
                _ => None,
            }
        }

        pub fn attribute(&self) -> Option<AssignmentValueAttribute> {
            match self.0.clone() {
                xc3_model::material::assignments::AssignmentValue::Attribute { name, channel } => {
                    Some(AssignmentValueAttribute {
                        name: name.to_string(),
                        channel,
                    })
                }
                _ => None,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::assignments::OutputAssignmentXyz)]
    pub struct OutputAssignmentXyz {
        pub assignment: usize,
        pub assignments: TypedList<AssignmentXyz>,
    }

    #[pyclass]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::assignments::AssignmentXyz)]
    pub struct AssignmentXyz(pub xc3_model::material::assignments::AssignmentXyz);

    #[pyclass]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::assignments::AssignmentValueXyz)]
    pub struct AssignmentValueXyz(pub xc3_model::material::assignments::AssignmentValueXyz);

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct AssignmentFuncXyz {
        pub op: Operation,
        pub args: Vec<usize>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct TextureAssignmentXyz {
        pub name: String,
        pub channel: Option<ChannelXyz>,
        pub texcoords: Vec<usize>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct AssignmentValueAttributeXyz {
        pub name: String,
        pub channel: Option<ChannelXyz>,
    }

    #[pymodule_export]
    use super::ChannelXyz;

    #[pymethods]
    impl AssignmentXyz {
        pub fn func(&self) -> Option<AssignmentFuncXyz> {
            match &self.0 {
                xc3_model::material::assignments::AssignmentXyz::Func { op, args } => {
                    Some(AssignmentFuncXyz {
                        op: (*op).into(),
                        args: args.clone(),
                    })
                }
                _ => None,
            }
        }

        pub fn value(&self) -> Option<AssignmentValueXyz> {
            match &self.0 {
                xc3_model::material::assignments::AssignmentXyz::Value(v) => {
                    Some(AssignmentValueXyz(v.clone()?))
                }
                _ => None,
            }
        }
    }

    #[pymethods]
    impl AssignmentValueXyz {
        pub fn texture(&self) -> Option<TextureAssignmentXyz> {
            match &self.0 {
                xc3_model::material::assignments::AssignmentValueXyz::Texture(t) => {
                    Some(TextureAssignmentXyz {
                        name: t.name.to_string(),
                        channel: t.channel.map(Into::into),
                        texcoords: t.texcoords.clone(),
                    })
                }
                _ => None,
            }
        }

        pub fn float(&self) -> Option<(f32, f32, f32)> {
            match self.0 {
                xc3_model::material::assignments::AssignmentValueXyz::Float(f) => {
                    Some((f[0].0, f[1].0, f[2].0))
                }
                _ => None,
            }
        }

        pub fn attribute(&self) -> Option<AssignmentValueAttributeXyz> {
            match self.0.clone() {
                xc3_model::material::assignments::AssignmentValueXyz::Attribute {
                    name,
                    channel,
                } => Some(AssignmentValueAttributeXyz {
                    name: name.to_string(),
                    channel: channel.map(Into::into),
                }),
                _ => None,
            }
        }
    }
}
