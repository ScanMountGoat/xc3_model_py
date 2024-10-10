use pyo3::prelude::*;

use crate::{map_py_into_impl, map_py_wrapper_impl, python_enum, MapPy};

map_py_into_impl!(xc3_model::material::MaterialFlags, u32);
map_py_into_impl!(xc3_model::material::MaterialRenderFlags, u32);

map_py_wrapper_impl!(
    xc3_model::material::ChannelAssignment,
    material::ChannelAssignment
);

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
    Unk41,
    Unk49,
    Unk97,
    Unk105
);

python_enum!(
    StencilMode,
    xc3_model::material::StencilMode,
    Unk0,
    Unk1,
    Unk2,
    Unk6,
    Unk7,
    Unk8
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
    Unk16
);

python_enum!(
    RenderPassType,
    xc3_model::material::RenderPassType,
    Unk0,
    Unk1,
    Unk6,
    Unk7,
    Unk9
);

#[pymodule]
pub mod material {
    use crate::shader_database::{shader_database::ShaderProgram, LayerBlendMode};
    use crate::{map_py::MapPy, xc3_model_py::ImageTexture};
    use pyo3::prelude::*;
    use pyo3::types::PyList;

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
        // TODO: how to handle flags?
        pub flags: u32,
        pub render_flags: u32,
        pub state_flags: StateFlags,
        pub color: [f32; 4],
        pub textures: Py<PyList>,
        pub alpha_test: Option<TextureAlphaTest>,
        pub work_values: Vec<f32>,
        pub shader_vars: Vec<(u16, u16)>,
        pub work_callbacks: Py<PyList>,
        pub alpha_test_ref: [u8; 4],
        pub m_unks1_1: u32,
        pub m_unks1_2: u32,
        pub m_unks1_3: u32,
        pub m_unks1_4: u32,
        pub shader: Option<ShaderProgram>,
        pub technique_index: usize,
        pub pass_type: RenderPassType,
        pub parameters: MaterialParameters,
        pub m_unks2_2: u16,
        pub m_unks3_1: u16,
        pub fur_params: Option<FurShellParams>,
    }

    #[pymethods]
    impl Material {
        #[new]
        fn new(
            name: String,
            flags: u32,
            render_flags: u32,
            state_flags: StateFlags,
            color: [f32; 4],
            textures: Py<PyList>,
            work_values: Vec<f32>,
            shader_vars: Vec<(u16, u16)>,
            work_callbacks: Py<PyList>,
            alpha_test_ref: [u8; 4],
            m_unks1_1: u32,
            m_unks1_2: u32,
            m_unks1_3: u32,
            m_unks1_4: u32,
            technique_index: usize,
            pass_type: RenderPassType,
            parameters: MaterialParameters,
            m_unks2_2: u16,
            m_unks3_1: u16,
            alpha_test: Option<TextureAlphaTest>,
            shader: Option<ShaderProgram>,
            fur_params: Option<FurShellParams>,
        ) -> Self {
            Self {
                name,
                flags,
                render_flags,
                state_flags,
                color,
                textures,
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
                m_unks2_2,
                m_unks3_1,
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

            let assignments: xc3_model::material::OutputAssignments =
                self.map_py(py)?.output_assignments(&image_textures);
            assignments.map_py(py)
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
    #[map(xc3_model::material::TextureAlphaTest)]
    pub struct TextureAlphaTest {
        pub texture_index: usize,
        pub channel_index: usize,
    }

    #[pymethods]
    impl TextureAlphaTest {
        #[new]
        fn new(texture_index: usize, channel_index: usize) -> Self {
            Self {
                texture_index,
                channel_index,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::MaterialParameters)]
    pub struct MaterialParameters {
        pub alpha_test_ref: f32,
        pub tex_matrix: Option<Vec<[f32; 8]>>,
        pub work_float4: Option<Vec<[f32; 4]>>,
        pub work_color: Option<Vec<[f32; 4]>>,
    }

    #[pymethods]
    impl MaterialParameters {
        #[new]
        fn new(
            alpha_test_ref: f32,
            tex_matrix: Option<Vec<[f32; 8]>>,
            work_float4: Option<Vec<[f32; 4]>>,
            work_color: Option<Vec<[f32; 4]>>,
        ) -> Self {
            Self {
                alpha_test_ref,
                tex_matrix,
                work_float4,
                work_color,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::Texture)]
    pub struct Texture {
        pub image_texture_index: usize,
        pub sampler_index: usize,
    }

    #[pymethods]
    impl Texture {
        #[new]
        fn new(image_texture_index: usize, sampler_index: usize) -> Self {
            Self {
                image_texture_index,
                sampler_index,
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
        pub unk2: f32,
        pub unk3: f32,
        pub unk4: f32,
        pub unk5: f32,
    }

    #[pymethods]
    impl FurShellParams {
        #[new]
        fn new(instance_count: u32, unk2: f32, unk3: f32, unk4: f32, unk5: f32) -> Self {
            Self {
                instance_count,
                unk2,
                unk3,
                unk4,
                unk5,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::OutputAssignments)]
    pub struct OutputAssignments {
        pub assignments: [OutputAssignment; 6],
    }

    #[pymethods]
    impl OutputAssignments {
        fn mat_id(&self, py: Python) -> PyResult<Option<u32>> {
            let assignments: xc3_model::material::OutputAssignments = self.map_py(py)?;
            Ok(assignments.mat_id())
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::OutputAssignment)]
    pub struct OutputAssignment {
        pub x: Option<ChannelAssignment>,
        pub y: Option<ChannelAssignment>,
        pub z: Option<ChannelAssignment>,
        pub w: Option<ChannelAssignment>,
        pub layers: Py<PyList>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::material::OutputLayerAssignment)]
    pub struct OutputLayerAssignment {
        pub x: Option<ChannelAssignment>,
        pub y: Option<ChannelAssignment>,
        pub z: Option<ChannelAssignment>,
        pub w: Option<ChannelAssignment>,
        pub weight: Option<ChannelAssignment>,
        pub blend_mode: LayerBlendMode,
        pub is_fresnel: bool,
    }

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct ChannelAssignment(pub xc3_model::material::ChannelAssignment);

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct TextureAssignment {
        pub name: String,
        pub channels: String,
        pub texcoord_name: Option<String>,
        pub texcoord_transforms: Option<((f32, f32, f32, f32), (f32, f32, f32, f32))>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct ChannelAssignmentAttribute {
        pub name: String,
        pub channel_index: usize,
    }

    #[pymethods]
    impl ChannelAssignment {
        // Workaround for representing Rust enums in Python.
        pub fn texture(&self) -> Option<TextureAssignment> {
            match &self.0 {
                xc3_model::material::ChannelAssignment::Texture(t) => Some(TextureAssignment {
                    name: t.name.to_string(),
                    channels: t.channels.to_string(),
                    texcoord_name: t.texcoord_name.as_ref().map(|s| s.to_string()),
                    texcoord_transforms: t.texcoord_transforms.map(|(u, v)| (u.into(), v.into())),
                }),
                _ => None,
            }
        }

        pub fn value(&self) -> Option<f32> {
            match self.0 {
                xc3_model::material::ChannelAssignment::Value(f) => Some(f),
                _ => None,
            }
        }

        pub fn attribute(&self) -> Option<ChannelAssignmentAttribute> {
            match self.0.clone() {
                xc3_model::material::ChannelAssignment::Attribute {
                    name,
                    channel_index,
                } => Some(ChannelAssignmentAttribute {
                    name: name.to_string(),
                    channel_index,
                }),
                _ => None,
            }
        }
    }
}
