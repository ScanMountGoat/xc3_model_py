use std::{ops::Deref, path::Path};

use crate::map_py::MapPy;
use glam::Mat4;
use numpy::{IntoPyArray, PyArray, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::{create_exception, exceptions::PyException, prelude::*, types::PyList};
use rayon::prelude::*;
use shader_database::{ShaderDatabase, ShaderProgram};
use vertex::ModelBuffers;

mod animation;
mod map_py;
mod shader_database;
mod skinning;
mod vertex;

// Create a Python class for every public xc3_model type.
// We don't define these conversions on the xc3_model types themselves.
// This flexibility allows more efficient and idiomatic bindings.
// Py<PyList> creates a pure Python list for allowing mutating of elements.
// PyObject supports numpy.ndarray for performance.
// Vec<u8> uses Python's bytes object for performance.

create_exception!(xc3_model_py, Xc3ModelError, PyException);

#[macro_export]
macro_rules! python_enum {
    ($py_ty:ident, $rust_ty:ty, $( $i:ident ),+) => {
        #[pyclass]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $py_ty {
            $($i),*
        }

        // These will generate a compile error if variant names don't match.
        impl From<$rust_ty> for $py_ty {
            fn from(value: $rust_ty) -> Self {
                match value {
                    $(<$rust_ty>::$i => Self::$i),*
                }
            }
        }

        impl From<$py_ty> for $rust_ty {
            fn from(value: $py_ty) -> Self {
                match value {
                    $(<$py_ty>::$i => Self::$i),*
                }
            }
        }

        impl $crate::map_py::MapPy<$rust_ty> for $py_ty {
            fn map_py(&self, _py: Python) -> PyResult<$rust_ty> {
                Ok((*self).into())
            }
        }

        impl $crate::map_py::MapPy<$py_ty> for $rust_ty {
            fn map_py(&self, _py: Python) -> PyResult<$py_ty> {
                Ok((*self).into())
            }
        }
    };
}

// TODO: traits and functions to simplify list conversions?

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::ModelRoot)]
pub struct ModelRoot {
    pub models: Py<Models>,
    pub buffers: Py<ModelBuffers>,
    pub image_textures: Py<PyList>,
    pub skeleton: Option<Py<Skeleton>>,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::MapRoot)]
pub struct MapRoot {
    pub groups: Py<PyList>,
    pub image_textures: Py<PyList>,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::ModelGroup)]
pub struct ModelGroup {
    pub models: Py<PyList>,
    pub buffers: Py<PyList>,
}

#[pymethods]
impl ModelGroup {
    #[new]
    pub fn new(models: Py<PyList>, buffers: Py<PyList>) -> Self {
        Self { models, buffers }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::Models)]
pub struct Models {
    pub models: Py<PyList>,
    pub materials: Py<PyList>,
    pub samplers: Py<PyList>,
    pub morph_controller_names: Py<PyList>,
    pub animation_morph_names: Py<PyList>,
    pub max_xyz: [f32; 3],
    pub min_xyz: [f32; 3],
    pub lod_data: Option<Py<LodData>>,
}

#[pymethods]
impl Models {
    #[new]
    pub fn new(
        models: Py<PyList>,
        materials: Py<PyList>,
        samplers: Py<PyList>,
        max_xyz: [f32; 3],
        min_xyz: [f32; 3],
        morph_controller_names: Py<PyList>,
        animation_morph_names: Py<PyList>,
        lod_data: Option<Py<LodData>>,
    ) -> Self {
        Self {
            models,
            materials,
            samplers,
            lod_data,
            morph_controller_names,
            animation_morph_names,
            max_xyz,
            min_xyz,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::Model)]
pub struct Model {
    pub meshes: Py<PyList>,
    // N x 4 x 4 numpy.ndarray
    pub instances: PyObject,
    pub model_buffers_index: usize,
    pub max_xyz: [f32; 3],
    pub min_xyz: [f32; 3],
    pub bounding_radius: f32,
}

#[pymethods]
impl Model {
    #[new]
    pub fn new(
        meshes: Py<PyList>,
        instances: PyObject,
        model_buffers_index: usize,
        max_xyz: [f32; 3],
        min_xyz: [f32; 3],
        bounding_radius: f32,
    ) -> Self {
        Self {
            meshes,
            instances,
            model_buffers_index,
            max_xyz,
            min_xyz,
            bounding_radius,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::Mesh)]
pub struct Mesh {
    pub vertex_buffer_index: usize,
    pub index_buffer_index: usize,
    pub index_buffer_index2: usize,
    pub material_index: usize,
    pub ext_mesh_index: Option<usize>,
    pub lod_item_index: Option<usize>,
    pub flags1: u32,
    pub flags2: u32,
    pub base_mesh_index: Option<usize>,
}

#[pymethods]
impl Mesh {
    #[new]
    pub fn new(
        vertex_buffer_index: usize,
        index_buffer_index: usize,
        index_buffer_index2: usize,
        material_index: usize,
        flags1: u32,
        flags2: u32,
        lod_item_index: Option<usize>,
        ext_mesh_index: Option<usize>,
        base_mesh_index: Option<usize>,
    ) -> Self {
        Self {
            vertex_buffer_index,
            index_buffer_index,
            index_buffer_index2,
            material_index,
            ext_mesh_index,
            lod_item_index,
            flags1,
            flags2,
            base_mesh_index,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::LodData)]
pub struct LodData {
    pub items: Py<PyList>,
    pub groups: Py<PyList>,
}

#[pymethods]
impl LodData {
    #[new]
    pub fn new(items: Py<PyList>, groups: Py<PyList>) -> Self {
        Self { items, groups }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::LodItem)]
pub struct LodItem {
    pub unk2: f32,
    pub index: u8,
    pub unk5: u8,
}

#[pymethods]
impl LodItem {
    #[new]
    pub fn new(unk2: f32, index: u8, unk5: u8) -> Self {
        Self { unk2, index, unk5 }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::LodGroup)]
pub struct LodGroup {
    pub base_lod_index: usize,
    pub lod_count: usize,
}

#[pymethods]
impl LodGroup {
    #[new]
    pub fn new(base_lod_index: usize, lod_count: usize) -> Self {
        Self {
            base_lod_index,
            lod_count,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::Skeleton)]
pub struct Skeleton {
    pub bones: Py<PyList>,
}

#[pymethods]
impl Skeleton {
    // TODO: generate this with some sort of macro?
    #[new]
    fn new(bones: Py<PyList>) -> Self {
        Self { bones }
    }

    pub fn model_space_transforms(&self, py: Python) -> PyResult<PyObject> {
        let transforms = self.map_py(py)?.model_space_transforms();
        Ok(transforms_pyarray(py, &transforms))
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::Bone)]
pub struct Bone {
    pub name: String,
    pub transform: PyObject,
    pub parent_index: Option<usize>,
}

#[pymethods]
impl Bone {
    #[new]
    fn new(name: String, transform: PyObject, parent_index: Option<usize>) -> Self {
        Self {
            name,
            transform,
            parent_index,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::Material)]
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
    pub work_callbacks: Vec<(u16, u16)>,
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
        work_callbacks: Vec<(u16, u16)>,
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

        let assignments = self.map_py(py)?.output_assignments(&image_textures);
        Ok(output_assignments_py(assignments))
    }
}

python_enum!(
    RenderPassType,
    xc3_model::RenderPassType,
    Unk0,
    Unk1,
    Unk6,
    Unk7,
    Unk9
);

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::StateFlags)]
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

python_enum!(
    BlendMode,
    xc3_model::BlendMode,
    Disabled,
    Blend,
    Unk2,
    Multiply,
    MultiplyInverted,
    Add,
    Disabled2
);

python_enum!(CullMode, xc3_model::CullMode, Back, Front, Disabled, Unk3);

python_enum!(
    StencilValue,
    xc3_model::StencilValue,
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
    xc3_model::StencilMode,
    Unk0,
    Unk1,
    Unk2,
    Unk6,
    Unk7,
    Unk8
);

python_enum!(DepthFunc, xc3_model::DepthFunc, Disabled, LessEqual, Equal);

python_enum!(
    ColorWriteMode,
    xc3_model::ColorWriteMode,
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

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::TextureAlphaTest)]
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
#[map(xc3_model::MaterialParameters)]
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
#[map(xc3_model::Texture)]
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

// TODO: MapPy won't work with threads?
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::ImageTexture)]
pub struct ImageTexture {
    pub name: Option<String>,
    pub usage: Option<TextureUsage>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub view_dimension: ViewDimension,
    pub image_format: ImageFormat,
    pub mipmap_count: u32,
    pub image_data: Vec<u8>,
}

#[pymethods]
impl ImageTexture {
    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
        width: u32,
        height: u32,
        depth: u32,
        view_dimension: ViewDimension,
        image_format: ImageFormat,
        mipmap_count: u32,
        image_data: Vec<u8>,
        name: Option<String>,
        usage: Option<TextureUsage>,
    ) -> Self {
        Self {
            name,
            usage,
            width,
            height,
            depth,
            view_dimension,
            image_format,
            mipmap_count,
            image_data,
        }
    }
}

// Helper types for enabling parallel encoding.
// TODO: Initialize this without copies?
#[pyclass(get_all, set_all)]
pub struct EncodeSurfaceRgba32FloatArgs {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub view_dimension: ViewDimension,
    pub image_format: ImageFormat,
    pub mipmaps: bool,
    pub data: PyObject,
    pub name: Option<String>,
    pub usage: Option<TextureUsage>,
}

#[pymethods]
impl EncodeSurfaceRgba32FloatArgs {
    #[new]
    fn new(
        width: u32,
        height: u32,
        depth: u32,
        view_dimension: ViewDimension,
        image_format: ImageFormat,
        mipmaps: bool,
        data: PyObject,
        name: Option<String>,
        usage: Option<TextureUsage>,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            view_dimension,
            image_format,
            mipmaps,
            data,
            name,
            usage,
        }
    }

    fn encode(&self, py: Python) -> PyResult<ImageTexture> {
        let surface = self.to_surface(py)?;

        let format: xc3_model::ImageFormat = self.image_format.into();
        let encoded_surface = surface
            .encode(
                format.into(),
                image_dds::Quality::Normal,
                if self.mipmaps {
                    image_dds::Mipmaps::GeneratedAutomatic
                } else {
                    image_dds::Mipmaps::Disabled
                },
            )
            .map_err(py_exception)?;

        Ok(ImageTexture {
            name: self.name.clone(),
            usage: self.usage,
            width: self.width,
            height: self.height,
            depth: self.depth,
            view_dimension: self.view_dimension,
            image_format: self.image_format,
            mipmap_count: encoded_surface.mipmaps,
            image_data: encoded_surface.data,
        })
    }
}

impl EncodeSurfaceRgba32FloatArgs {
    fn to_surface(&self, py: Python) -> PyResult<image_dds::SurfaceRgba32Float<Vec<f32>>> {
        // Handle any dimensions but require the right data type.
        // Converting to a vec will "flatten" the array.
        let data: PyReadonlyArrayDyn<'_, f32> = self.data.extract(py)?;

        Ok(image_dds::SurfaceRgba32Float {
            width: self.width,
            height: self.height,
            depth: self.depth,
            layers: if self.view_dimension == ViewDimension::Cube {
                6
            } else {
                1
            },
            mipmaps: 1,
            data: data.to_vec()?,
        })
    }
}

#[pyclass(get_all, set_all)]
pub struct EncodeSurfaceRgba8Args {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub view_dimension: ViewDimension,
    pub image_format: ImageFormat,
    pub mipmaps: bool,
    pub data: Vec<u8>,
    pub name: Option<String>,
    pub usage: Option<TextureUsage>,
}

#[pymethods]
impl EncodeSurfaceRgba8Args {
    #[new]
    fn new(
        width: u32,
        height: u32,
        depth: u32,
        view_dimension: ViewDimension,
        image_format: ImageFormat,
        mipmaps: bool,
        data: Vec<u8>,
        name: Option<String>,
        usage: Option<TextureUsage>,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            view_dimension,
            image_format,
            mipmaps,
            data,
            name,
            usage,
        }
    }

    fn encode(&self) -> PyResult<ImageTexture> {
        let surface = self.as_surface();

        let format: xc3_model::ImageFormat = self.image_format.into();
        let encoded_surface = surface
            .encode(
                format.into(),
                image_dds::Quality::Normal,
                if self.mipmaps {
                    image_dds::Mipmaps::GeneratedAutomatic
                } else {
                    image_dds::Mipmaps::Disabled
                },
            )
            .map_err(py_exception)?;

        Ok(ImageTexture {
            name: self.name.clone(),
            usage: self.usage,
            width: self.width,
            height: self.height,
            depth: self.depth,
            view_dimension: self.view_dimension,
            image_format: self.image_format,
            mipmap_count: encoded_surface.mipmaps,
            image_data: encoded_surface.data,
        })
    }
}

impl EncodeSurfaceRgba8Args {
    fn as_surface(&self) -> image_dds::SurfaceRgba8<&[u8]> {
        image_dds::SurfaceRgba8 {
            width: self.width,
            height: self.height,
            depth: self.depth,
            layers: if self.view_dimension == ViewDimension::Cube {
                6
            } else {
                1
            },
            mipmaps: 1,
            data: &self.data,
        }
    }
}

python_enum!(
    TextureUsage,
    xc3_model::TextureUsage,
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

python_enum!(ViewDimension, xc3_model::ViewDimension, D2, D3, Cube);

python_enum!(
    ImageFormat,
    xc3_model::ImageFormat,
    R8Unorm,
    R8G8B8A8Unorm,
    R16G16B16A16Float,
    R4G4B4A4Unorm,
    BC1Unorm,
    BC2Unorm,
    BC3Unorm,
    BC4Unorm,
    BC5Unorm,
    BC6UFloat,
    BC7Unorm,
    B8G8R8A8Unorm
);

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::Sampler)]
pub struct Sampler {
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mip_filter: FilterMode,
    pub mipmaps: bool,
}

#[pymethods]
impl Sampler {
    #[new]
    fn new(
        address_mode_u: AddressMode,
        address_mode_v: AddressMode,
        address_mode_w: AddressMode,
        min_filter: FilterMode,
        mag_filter: FilterMode,
        mip_filter: FilterMode,
        mipmaps: bool,
    ) -> Self {
        Self {
            address_mode_u,
            address_mode_v,
            address_mode_w,
            min_filter,
            mag_filter,
            mip_filter,
            mipmaps,
        }
    }
}

python_enum!(
    AddressMode,
    xc3_model::AddressMode,
    ClampToEdge,
    Repeat,
    MirrorRepeat
);

python_enum!(FilterMode, xc3_model::FilterMode, Nearest, Linear);

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct OutputAssignments {
    pub assignments: [OutputAssignment; 6],
}

#[pymethods]
impl OutputAssignments {
    fn mat_id(&self) -> Option<u32> {
        output_assignments_rs(self).mat_id()
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct OutputAssignment {
    pub x: Option<ChannelAssignment>,
    pub y: Option<ChannelAssignment>,
    pub z: Option<ChannelAssignment>,
    pub w: Option<ChannelAssignment>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ChannelAssignment(xc3_model::ChannelAssignment);

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

#[pyclass]
#[derive(Debug, Clone)]
pub struct Mxmd(xc3_lib::mxmd::Mxmd);

#[pymethods]
impl Mxmd {
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        xc3_lib::mxmd::Mxmd::from_file(path)
            .map(Mxmd)
            .map_err(py_exception)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        self.0.save(path).map_err(py_exception)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Msrd(xc3_lib::msrd::Msrd);

#[pymethods]
impl Msrd {
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        xc3_lib::msrd::Msrd::from_file(path)
            .map(Msrd)
            .map_err(py_exception)
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        self.0.save(path).map_err(py_exception)
    }
}

#[pymethods]
impl ModelRoot {
    #[new]
    pub fn new(
        models: Py<Models>,
        buffers: Py<ModelBuffers>,
        image_textures: Py<PyList>,
        skeleton: Option<Py<Skeleton>>,
    ) -> Self {
        Self {
            models,
            buffers,
            image_textures,
            skeleton,
        }
    }

    pub fn save_images_rgba8(
        &self,
        py: Python,
        folder: &str,
        prefix: &str,
        ext: &str,
        flip_vertical: bool,
    ) -> PyResult<Vec<String>> {
        save_images_rgba8(py, &self.image_textures, folder, prefix, ext, flip_vertical)
    }

    pub fn to_mxmd_model(&self, py: Python, mxmd: &Mxmd, msrd: &Msrd) -> PyResult<(Mxmd, Msrd)> {
        let (mxmd, msrd) = self.map_py(py)?.to_mxmd_model(&mxmd.0, &msrd.0);
        Ok((Mxmd(mxmd), Msrd(msrd)))
    }
    // TODO: support texture edits as well?
}

#[pymethods]
impl MapRoot {
    #[new]
    pub fn new(groups: Py<PyList>, image_textures: Py<PyList>) -> Self {
        Self {
            groups,
            image_textures,
        }
    }

    pub fn save_images_rgba8(
        &self,
        py: Python,
        folder: &str,
        prefix: &str,
        ext: &str,
        flip_vertical: bool,
    ) -> PyResult<Vec<String>> {
        save_images_rgba8(py, &self.image_textures, folder, prefix, ext, flip_vertical)
    }
    // TODO: support texture edits as well?
}

fn save_images_rgba8(
    py: Python,
    image_textures: &Py<PyList>,
    folder: &str,
    prefix: &str,
    ext: &str,
    flip_vertical: bool,
) -> PyResult<Vec<String>> {
    let textures: Vec<xc3_model::ImageTexture> = image_textures.map_py(py)?;
    textures
        .par_iter()
        .enumerate()
        .map(|(i, texture)| {
            // Use the same naming conventions as xc3_tex.
            let filename = texture
                .name
                .as_ref()
                .map(|n| format!("{prefix}.{i}.{n}.{ext}"))
                .unwrap_or_else(|| format!("{prefix}.{i}.{ext}"));
            let path = Path::new(folder).join(filename);

            let mut image = texture.to_image().map_err(py_exception)?;
            if flip_vertical {
                // Xenoblade X images need to be flipped vertically to look as expected.
                // TODO: Is there a better way of handling this?
                image = image_dds::image::imageops::flip_vertical(&image);
            }

            image.save(&path).map_err(py_exception)?;

            Ok(path.to_string_lossy().to_string())
        })
        .collect()
}

#[pymethods]
impl ChannelAssignment {
    // Workaround for representing Rust enums in Python.
    pub fn textures(&self) -> Option<Vec<TextureAssignment>> {
        match &self.0 {
            xc3_model::ChannelAssignment::Textures(textures) => Some(
                textures
                    .iter()
                    .map(|t| TextureAssignment {
                        name: t.name.to_string(),
                        channels: t.channels.to_string(),
                        texcoord_name: t.texcoord_name.as_ref().map(|s| s.to_string()),
                        texcoord_transforms: t
                            .texcoord_transforms
                            .map(|(u, v)| (u.into(), v.into())),
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn value(&self) -> Option<f32> {
        match self.0 {
            xc3_model::ChannelAssignment::Value(f) => Some(f),
            _ => None,
        }
    }

    pub fn attribute(&self) -> Option<ChannelAssignmentAttribute> {
        match self.0.clone() {
            xc3_model::ChannelAssignment::Attribute {
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

#[pyfunction]
fn load_model(
    py: Python,
    wimdo_path: &str,
    shader_database: Option<&ShaderDatabase>,
) -> PyResult<ModelRoot> {
    let database = shader_database.map(|database| &database.0);
    let root = xc3_model::load_model(wimdo_path, database).map_err(py_exception)?;
    root.map_py(py)
}

#[pyfunction]
fn load_model_legacy(
    py: Python,
    camdo_path: &str,
    shader_database: Option<&ShaderDatabase>,
) -> PyResult<ModelRoot> {
    let database = shader_database.map(|database| &database.0);
    let root = xc3_model::load_model_legacy(camdo_path, database).map_err(py_exception)?;
    root.map_py(py)
}

#[pyfunction]
fn load_map(
    py: Python,
    wismhd_path: &str,
    shader_database: Option<&ShaderDatabase>,
) -> PyResult<Py<PyList>> {
    let database = shader_database.map(|database| &database.0);
    // Prevent Python from locking up while Rust processes map data in parallel.
    let roots =
        py.allow_threads(move || xc3_model::load_map(wismhd_path, database).map_err(py_exception))?;
    roots.map_py(py)
}

#[pyfunction]
fn load_animations(_py: Python, anim_path: &str) -> PyResult<Vec<animation::Animation>> {
    let animations = xc3_model::load_animations(anim_path).map_err(py_exception)?;
    Ok(animations
        .into_iter()
        .map(animation::animation_py)
        .collect())
}

#[pyfunction]
fn decode_images_rgbaf32(
    py: Python,
    image_textures: Vec<PyRef<ImageTexture>>,
) -> PyResult<Vec<PyObject>> {
    let textures: Vec<&ImageTexture> = image_textures.iter().map(|i| i.deref()).collect();
    let buffers = textures
        .par_iter()
        .map(|image| {
            let format: xc3_model::ImageFormat = image.image_format.into();
            let surface = image_dds::Surface {
                width: image.width,
                height: image.height,
                depth: image.depth,
                layers: if image.view_dimension == ViewDimension::Cube {
                    6
                } else {
                    1
                },
                mipmaps: image.mipmap_count,
                image_format: format.into(),
                data: &image.image_data,
            };

            Ok(surface
                .decode_layers_mipmaps_rgbaf32(0..surface.layers, 0..1)
                .map_err(py_exception)?
                .data)
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(buffers
        .into_iter()
        .map(|buffer| buffer.into_pyarray_bound(py).into())
        .collect())
}

#[pyfunction]
fn encode_images_rgba8(
    py: Python,
    images: Vec<PyRef<EncodeSurfaceRgba8Args>>,
) -> PyResult<Vec<ImageTexture>> {
    let surfaces: Vec<_> = images
        .iter()
        .map(|image| {
            (
                image.name.clone(),
                image.usage,
                image.image_format,
                image.mipmaps,
                image.as_surface(),
            )
        })
        .collect();

    // Prevent Python from locking up while Rust processes data in parallel.
    py.allow_threads(move || {
        surfaces
            .into_par_iter()
            .map(|(name, usage, image_format, mipmaps, surface)| {
                // TODO: quality?
                let format: xc3_model::ImageFormat = image_format.into();
                let encoded_surface = surface
                    .encode(
                        format.into(),
                        image_dds::Quality::Normal,
                        if mipmaps {
                            image_dds::Mipmaps::GeneratedAutomatic
                        } else {
                            image_dds::Mipmaps::Disabled
                        },
                    )
                    .map_err(py_exception)?;

                Ok(ImageTexture {
                    name,
                    usage,
                    width: surface.width,
                    height: surface.height,
                    depth: surface.depth,
                    view_dimension: if surface.layers == 6 {
                        ViewDimension::Cube
                    } else {
                        ViewDimension::D2
                    },
                    image_format,
                    mipmap_count: encoded_surface.mipmaps,
                    image_data: encoded_surface.data,
                })
            })
            .collect()
    })
}

#[pyfunction]
fn encode_images_rgbaf32(
    py: Python,
    images: Vec<PyRef<EncodeSurfaceRgba32FloatArgs>>,
) -> PyResult<Vec<ImageTexture>> {
    let surfaces = images
        .iter()
        .map(|image| {
            Ok((
                image.name.clone(),
                image.usage,
                image.image_format,
                image.mipmaps,
                image.to_surface(py)?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Prevent Python from locking up while Rust processes data in parallel.
    py.allow_threads(move || {
        surfaces
            .into_par_iter()
            .map(|(name, usage, image_format, mipmaps, surface)| {
                // TODO: quality?
                let format: xc3_model::ImageFormat = image_format.into();
                let encoded_surface = surface
                    .encode(
                        format.into(),
                        image_dds::Quality::Normal,
                        if mipmaps {
                            image_dds::Mipmaps::GeneratedAutomatic
                        } else {
                            image_dds::Mipmaps::Disabled
                        },
                    )
                    .map_err(py_exception)?;

                Ok(ImageTexture {
                    name,
                    usage,
                    width: surface.width,
                    height: surface.height,
                    depth: surface.depth,
                    view_dimension: if surface.layers == 6 {
                        ViewDimension::Cube
                    } else {
                        ViewDimension::D2
                    },
                    image_format,
                    mipmap_count: encoded_surface.mipmaps,
                    image_data: encoded_surface.data,
                })
            })
            .collect()
    })
}

fn py_exception<E: Into<anyhow::Error>>(e: E) -> PyErr {
    // anyhow provides more detailed context for inner errors.
    PyErr::new::<Xc3ModelError, _>(format!("{:?}", anyhow::anyhow!(e)))
}

// TODO: Create a proper type for this.
impl MapPy<xc3_model::MeshRenderFlags2> for u32 {
    fn map_py(&self, _py: Python) -> PyResult<xc3_model::MeshRenderFlags2> {
        Ok((*self).try_into().unwrap())
    }
}

impl MapPy<u32> for xc3_model::MeshRenderFlags2 {
    fn map_py(&self, _py: Python) -> PyResult<u32> {
        Ok((*self).into())
    }
}

fn output_assignments_py(assignments: xc3_model::OutputAssignments) -> OutputAssignments {
    OutputAssignments {
        assignments: assignments.assignments.map(|a| OutputAssignment {
            x: a.x.map(ChannelAssignment),
            y: a.y.map(ChannelAssignment),
            z: a.z.map(ChannelAssignment),
            w: a.w.map(ChannelAssignment),
        }),
    }
}

fn output_assignments_rs(assignments: &OutputAssignments) -> xc3_model::OutputAssignments {
    xc3_model::OutputAssignments {
        assignments: assignments
            .assignments
            .clone()
            .map(|a| xc3_model::OutputAssignment {
                x: a.x.map(|v| v.0),
                y: a.y.map(|v| v.0),
                z: a.z.map(|v| v.0),
                w: a.w.map(|v| v.0),
            }),
    }
}

fn uvec2s_pyarray(py: Python, values: &[[u16; 2]]) -> PyObject {
    // This flatten will be optimized in Release mode.
    // This avoids needing unsafe code.
    let count = values.len();
    values
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<u16>>()
        .into_pyarray_bound(py)
        .reshape((count, 2))
        .unwrap()
        .into()
}

fn uvec4_pyarray(py: Python, values: &[[u8; 4]]) -> PyObject {
    // This flatten will be optimized in Release mode.
    // This avoids needing unsafe code.
    let count = values.len();
    values
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<u8>>()
        .into_pyarray_bound(py)
        .reshape((count, 4))
        .unwrap()
        .into()
}

fn transforms_pyarray(py: Python, transforms: &[Mat4]) -> PyObject {
    // This flatten will be optimized in Release mode.
    // This avoids needing unsafe code.
    let transform_count = transforms.len();
    transforms
        .iter()
        .flat_map(|v| v.to_cols_array())
        .collect::<Vec<f32>>()
        .into_pyarray_bound(py)
        .reshape((transform_count, 4, 4))
        .unwrap()
        .into()
}

// TODO: test cases for conversions.
fn mat4_to_pyarray(py: Python, transform: Mat4) -> PyObject {
    PyArray::from_slice_bound(py, &transform.to_cols_array())
        .readwrite()
        .reshape((4, 4))
        .unwrap()
        .into()
}

fn pyarray_to_mat4(py: Python, transform: &PyObject) -> PyResult<Mat4> {
    let cols: [[f32; 4]; 4] = transform.extract(py)?;
    Ok(Mat4::from_cols_array_2d(&cols))
}

fn pyarray_to_mat4s(py: Python, values: &PyObject) -> PyResult<Vec<Mat4>> {
    let transforms: Vec<[[f32; 4]; 4]> = values.extract(py)?;
    Ok(transforms.iter().map(Mat4::from_cols_array_2d).collect())
}

map_py_into_impl!(xc3_model::MaterialFlags, u32);
map_py_into_impl!(xc3_model::MaterialRenderFlags, u32);

#[pymodule]
fn xc3_model_py(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    // Match the module hierarchy and types of xc3_model as closely as possible.
    animation::animation(py, m)?;
    shader_database::shader_database(py, m)?;
    skinning::skinning(py, m)?;
    vertex::vertex(py, m)?;

    m.add_class::<ModelRoot>()?;
    m.add_class::<MapRoot>()?;
    m.add_class::<ModelGroup>()?;

    m.add_class::<Models>()?;
    m.add_class::<Model>()?;
    m.add_class::<Mesh>()?;

    m.add_class::<LodData>()?;
    m.add_class::<LodItem>()?;
    m.add_class::<LodGroup>()?;

    m.add_class::<Skeleton>()?;
    m.add_class::<Bone>()?;

    m.add_class::<Material>()?;
    m.add_class::<TextureAlphaTest>()?;
    m.add_class::<MaterialParameters>()?;
    m.add_class::<RenderPassType>()?;

    m.add_class::<StateFlags>()?;
    m.add_class::<BlendMode>()?;
    m.add_class::<CullMode>()?;
    m.add_class::<StencilValue>()?;
    m.add_class::<StencilMode>()?;
    m.add_class::<DepthFunc>()?;
    m.add_class::<ColorWriteMode>()?;

    m.add_class::<Texture>()?;

    m.add_class::<ImageTexture>()?;
    m.add_class::<TextureUsage>()?;
    m.add_class::<ViewDimension>()?;
    m.add_class::<ImageFormat>()?;
    m.add_class::<EncodeSurfaceRgba32FloatArgs>()?;
    m.add_class::<EncodeSurfaceRgba8Args>()?;

    m.add_class::<Sampler>()?;
    m.add_class::<AddressMode>()?;
    m.add_class::<FilterMode>()?;

    m.add_class::<OutputAssignments>()?;
    m.add_class::<OutputAssignment>()?;
    m.add_class::<ChannelAssignment>()?;
    m.add_class::<TextureAssignment>()?;

    m.add_class::<Mxmd>()?;
    m.add_class::<Msrd>()?;

    m.add("Xc3ModelError", py.get_type_bound::<Xc3ModelError>())?;

    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(load_model_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(load_map, m)?)?;
    m.add_function(wrap_pyfunction!(load_animations, m)?)?;
    m.add_function(wrap_pyfunction!(decode_images_rgbaf32, m)?)?;
    m.add_function(wrap_pyfunction!(encode_images_rgba8, m)?)?;
    m.add_function(wrap_pyfunction!(encode_images_rgbaf32, m)?)?;

    Ok(())
}
