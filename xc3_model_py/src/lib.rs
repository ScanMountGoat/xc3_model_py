use std::path::Path;

use crate::map_py::MapPy;
use glam::Mat4;
use numpy::{IntoPyArray, PyArray, PyArrayMethods};
use pyo3::{create_exception, exceptions::PyException, prelude::*, types::PyList};
use rayon::prelude::*;
use vertex::{model_buffers_py, model_buffers_rs, ModelBuffers};

mod animation;
mod map_py;
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
        #[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone)]
pub struct ModelRoot {
    pub models: Py<Models>,
    pub buffers: Py<ModelBuffers>,
    pub image_textures: Py<PyList>,
    pub skeleton: Option<Skeleton>,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct MapRoot {
    pub groups: Py<PyList>,
    pub image_textures: Py<PyList>,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct Models {
    pub models: Py<PyList>,
    pub materials: Py<PyList>,
    pub samplers: Py<PyList>,
    pub morph_controller_names: Py<PyList>,
    pub animation_morph_names: Py<PyList>,
    pub max_xyz: [f32; 3],
    pub min_xyz: [f32; 3],
    pub lod_data: Option<LodData>,
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
        lod_data: Option<LodData>,
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
        let transforms = skeleton_rs(py, self)?.model_space_transforms();
        Ok(transforms_pyarray(py, &transforms))
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    // TODO: how to handle flags?
    pub state_flags: StateFlags,
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
    pub shader: Option<Shader>,
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
        state_flags: StateFlags,
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
        shader: Option<Shader>,
    ) -> Self {
        Self {
            name,
            state_flags,
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

        let assignments = material_rs(py, self)?.output_assignments(&image_textures);
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
    pub color_write_mode: u8,
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

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct MaterialParameters {
    pub mat_color: [f32; 4],
    pub alpha_test_ref: f32,
    pub tex_matrix: Option<Vec<[f32; 8]>>,
    pub work_float4: Option<Vec<[f32; 4]>>,
    pub work_color: Option<Vec<[f32; 4]>>,
}

#[pymethods]
impl MaterialParameters {
    #[new]
    fn new(
        mat_color: [f32; 4],
        alpha_test_ref: f32,
        tex_matrix: Option<Vec<[f32; 8]>>,
        work_float4: Option<Vec<[f32; 4]>>,
        work_color: Option<Vec<[f32; 4]>>,
    ) -> Self {
        Self {
            mat_color,
            alpha_test_ref,
            tex_matrix,
            work_float4,
            work_color,
        }
    }
}

// TODO: Expose implementation details?
#[pyclass]
#[derive(Debug, Clone)]
pub struct Shader(xc3_model::shader_database::Shader);

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
pub struct ChannelAssignmentTexture {
    pub name: String,
    pub channel_index: usize,
    pub texcoord_name: Option<String>,
    pub texcoord_scale: Option<(f32, f32)>,
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
        skeleton: Option<Skeleton>,
    ) -> Self {
        Self {
            models,
            buffers,
            image_textures,
            skeleton,
        }
    }

    pub fn decode_images_rgbaf32(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let buffers = self
            .image_textures
            .extract::<'_, '_, Vec<ImageTexture>>(py)?
            .par_iter()
            .map(|image| {
                // TODO: Use image_dds directly to avoid cloning?
                let bytes = image_texture_rs(image)
                    .to_image()
                    .map_err(py_exception)?
                    .into_raw();

                Ok(bytes
                    .into_iter()
                    .map(|u| u as f32 / 255.0)
                    .collect::<Vec<_>>())
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(buffers
            .into_iter()
            .map(|buffer| buffer.into_pyarray_bound(py).into())
            .collect())
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
        let (mxmd, msrd) = model_root_rs(py, self)?.to_mxmd_model(&mxmd.0, &msrd.0);
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

    pub fn decode_images_rgbaf32(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let buffers = self
            .image_textures
            .extract::<'_, '_, Vec<ImageTexture>>(py)?
            .par_iter()
            .map(|image| {
                // TODO: Use image_dds directly to avoid cloning?
                let bytes = image_texture_rs(image)
                    .to_image()
                    .map_err(py_exception)?
                    .into_raw();

                Ok(bytes
                    .into_iter()
                    .map(|u| u as f32 / 255.0)
                    .collect::<Vec<_>>())
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(buffers
            .into_iter()
            .map(|buffer| buffer.into_pyarray_bound(py).into())
            .collect())
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
    image_textures
        .extract::<'_, '_, Vec<ImageTexture>>(py)?
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

            let mut image = image_texture_rs(texture).to_image().map_err(py_exception)?;
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
    pub fn texture(&self) -> Option<ChannelAssignmentTexture> {
        match self.0.clone() {
            xc3_model::ChannelAssignment::Texture {
                name,
                channel_index,
                texcoord_name,
                texcoord_scale,
            } => Some(ChannelAssignmentTexture {
                name,
                channel_index,
                texcoord_name,
                texcoord_scale,
            }),
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
                name,
                channel_index,
            }),
            _ => None,
        }
    }
}

#[pyfunction]
fn load_model(py: Python, wimdo_path: &str, database_path: Option<&str>) -> PyResult<ModelRoot> {
    let database = database_path
        .map(xc3_model::shader_database::ShaderDatabase::from_file)
        .transpose()
        .map_err(py_exception)?;
    let root = xc3_model::load_model(wimdo_path, database.as_ref()).map_err(py_exception)?;
    model_root_py(py, root)
}

#[pyfunction]
fn load_model_legacy(py: Python, camdo_path: &str) -> PyResult<ModelRoot> {
    let root = xc3_model::load_model_legacy(camdo_path).map_err(py_exception)?;
    model_root_py(py, root)
}

#[pyfunction]
fn load_map(py: Python, wismhd_path: &str, database_path: Option<&str>) -> PyResult<Vec<MapRoot>> {
    let database = database_path
        .map(xc3_model::shader_database::ShaderDatabase::from_file)
        .transpose()
        .map_err(py_exception)?;
    // Prevent Python from locking up while Rust processes map data in parallel.
    let roots = py.allow_threads(move || {
        xc3_model::load_map(wismhd_path, database.as_ref()).map_err(py_exception)
    })?;
    roots
        .into_iter()
        .map(|root| map_root_py(py, root))
        .collect()
}

#[pyfunction]
fn load_animations(_py: Python, anim_path: &str) -> PyResult<Vec<animation::Animation>> {
    let animations = xc3_model::load_animations(anim_path).map_err(py_exception)?;
    Ok(animations
        .into_iter()
        .map(animation::animation_py)
        .collect())
}

fn py_exception<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<Xc3ModelError, _>(format!("{e}"))
}

fn map_root_py(py: Python, root: xc3_model::MapRoot) -> PyResult<MapRoot> {
    // TODO: Avoid unwrap.
    Ok(MapRoot {
        groups: PyList::new_bound(
            py,
            root.groups
                .into_iter()
                .map(|group| model_group_py(py, group).unwrap().into_py(py)),
        )
        .into(),
        image_textures: PyList::new_bound(
            py,
            root.image_textures
                .into_iter()
                .map(|t| image_texture_py(t).into_py(py)),
        )
        .into(),
    })
}

fn model_root_py(py: Python, root: xc3_model::ModelRoot) -> PyResult<ModelRoot> {
    Ok(ModelRoot {
        models: Py::new(py, models_py(py, root.models))?,
        buffers: Py::new(py, model_buffers_py(py, root.buffers)?)?,
        image_textures: PyList::new_bound(
            py,
            root.image_textures
                .into_iter()
                .map(|t| image_texture_py(t).into_py(py)),
        )
        .into(),
        skeleton: root.skeleton.map(|skeleton| Skeleton {
            bones: PyList::new_bound(
                py,
                skeleton
                    .bones
                    .into_iter()
                    .map(|bone| bone_py(bone, py).into_py(py)),
            )
            .into(),
        }),
    })
}

fn models_py(py: Python, models: xc3_model::Models) -> Models {
    Models {
        models: PyList::new_bound(
            py,
            models
                .models
                .into_iter()
                .map(|m| model_py(py, m).into_py(py)),
        )
        .into(),
        materials: materials_py(py, models.materials),
        samplers: PyList::new_bound(
            py,
            models
                .samplers
                .iter()
                .map(|s| s.map_py(py).unwrap().into_py(py)),
        )
        .into(),
        lod_data: models.lod_data.map(|data| LodData {
            items: PyList::new_bound(
                py,
                data.items.into_iter().map(|i| {
                    LodItem {
                        unk2: i.unk2,
                        index: i.index,
                        unk5: i.unk5,
                    }
                    .into_py(py)
                }),
            )
            .into(),
            groups: PyList::new_bound(
                py,
                data.groups.into_iter().map(|g| {
                    LodGroup {
                        base_lod_index: g.base_lod_index,
                        lod_count: g.lod_count,
                    }
                    .into_py(py)
                }),
            )
            .into(),
        }),
        morph_controller_names: PyList::new_bound(py, models.morph_controller_names).into(),
        animation_morph_names: PyList::new_bound(py, models.animation_morph_names).into(),
        max_xyz: models.max_xyz.to_array(),
        min_xyz: models.min_xyz.to_array(),
    }
}

fn models_rs(py: Python, models: &Models) -> PyResult<xc3_model::Models> {
    Ok(xc3_model::Models {
        models: models
            .models
            .extract::<'_, '_, Vec<Model>>(py)?
            .iter()
            .map(|model| model_rs(py, model))
            .collect::<PyResult<Vec<_>>>()?,
        materials: models
            .materials
            .extract::<'_, '_, Vec<Material>>(py)?
            .iter()
            .map(|m| material_rs(py, m))
            .collect::<PyResult<Vec<_>>>()?,
        samplers: models
            .samplers
            .extract::<'_, '_, Vec<Sampler>>(py)?
            .iter()
            .map(|s| s.map_py(py).unwrap())
            .collect(),
        lod_data: models
            .lod_data
            .as_ref()
            .map(|data| lod_data_rs(data, py))
            .transpose()?,
        morph_controller_names: models.morph_controller_names.extract(py)?,
        animation_morph_names: models.animation_morph_names.extract(py)?,
        max_xyz: models.max_xyz.into(),
        min_xyz: models.min_xyz.into(),
    })
}

fn lod_data_rs(data: &LodData, py: Python) -> PyResult<xc3_model::LodData> {
    Ok(xc3_model::LodData {
        items: data
            .items
            .extract::<'_, '_, Vec<LodItem>>(py)?
            .iter()
            .map(|i| xc3_model::LodItem {
                unk2: i.unk2,
                index: i.index,
                unk5: i.unk5,
            })
            .collect(),
        groups: data
            .groups
            .extract::<'_, '_, Vec<LodGroup>>(py)?
            .iter()
            .map(|g| xc3_model::LodGroup {
                base_lod_index: g.base_lod_index,
                lod_count: g.lod_count,
            })
            .collect(),
    })
}

fn model_group_py(py: Python, group: xc3_model::ModelGroup) -> PyResult<ModelGroup> {
    // TODO: avoid unwrap.
    Ok(ModelGroup {
        models: PyList::new_bound(
            py,
            group
                .models
                .into_iter()
                .map(|models| models_py(py, models).into_py(py)),
        )
        .into(),
        buffers: PyList::new_bound(
            py,
            group
                .buffers
                .into_iter()
                .map(|buffer| model_buffers_py(py, buffer).unwrap().into_py(py)),
        )
        .into(),
    })
}

fn model_rs(py: Python, model: &Model) -> PyResult<xc3_model::Model> {
    Ok(xc3_model::Model {
        meshes: model
            .meshes
            .extract::<'_, '_, Vec<Mesh>>(py)?
            .iter()
            .map(|mesh| xc3_model::Mesh {
                vertex_buffer_index: mesh.vertex_buffer_index,
                index_buffer_index: mesh.index_buffer_index,
                index_buffer_index2: mesh.index_buffer_index2,
                material_index: mesh.material_index,
                ext_mesh_index: mesh.ext_mesh_index,
                lod_item_index: mesh.lod_item_index,
                flags1: mesh.flags1,
                flags2: mesh.flags2.try_into().unwrap(),
                base_mesh_index: mesh.base_mesh_index,
            })
            .collect(),
        instances: pyarray_to_mat4s(py, &model.instances)?,
        model_buffers_index: model.model_buffers_index,
        max_xyz: model.max_xyz.into(),
        min_xyz: model.min_xyz.into(),
        bounding_radius: model.bounding_radius,
    })
}

fn material_rs(py: Python, material: &Material) -> PyResult<xc3_model::Material> {
    // TODO: Properly support flags conversions.
    Ok(xc3_model::Material {
        name: material.name.clone(),
        flags: xc3_model::MaterialFlags::from(0u32),
        render_flags: xc3_model::MaterialRenderFlags::from(0u32),
        state_flags: material.state_flags.map_py(py)?,
        textures: material
            .textures
            .extract::<'_, '_, Vec<Texture>>(py)?
            .iter()
            .map(|t| xc3_model::Texture {
                image_texture_index: t.image_texture_index,
                sampler_index: t.sampler_index,
            })
            .collect(),
        alpha_test: material
            .alpha_test
            .as_ref()
            .map(|a| xc3_model::TextureAlphaTest {
                texture_index: a.texture_index,
                channel_index: a.channel_index,
            }),
        work_values: material.work_values.clone(),
        shader_vars: material.shader_vars.clone(),
        work_callbacks: material.work_callbacks.clone(),
        alpha_test_ref: material.alpha_test_ref,
        m_unks1_1: material.m_unks1_1,
        m_unks1_2: material.m_unks1_2,
        m_unks1_3: material.m_unks1_3,
        m_unks1_4: material.m_unks1_4,
        shader: material.shader.clone().map(|s| s.0),
        technique_index: material.technique_index,
        pass_type: material.pass_type.into(),
        parameters: xc3_model::MaterialParameters {
            mat_color: material.parameters.mat_color,
            alpha_test_ref: material.parameters.alpha_test_ref,
            tex_matrix: material.parameters.tex_matrix.clone(),
            work_float4: material.parameters.work_float4.clone(),
            work_color: material.parameters.work_color.clone(),
        },
        m_unks2_2: material.m_unks2_2,
        m_unks3_1: material.m_unks3_1,
    })
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

fn image_texture_py(image: xc3_model::ImageTexture) -> ImageTexture {
    ImageTexture {
        name: image.name,
        usage: image.usage.map(Into::into),
        width: image.width,
        height: image.height,
        depth: image.depth,
        view_dimension: image.view_dimension.into(),
        image_format: image.image_format.into(),
        mipmap_count: image.mipmap_count,
        image_data: image.image_data,
    }
}

fn image_texture_rs(image: &ImageTexture) -> xc3_model::ImageTexture {
    xc3_model::ImageTexture {
        name: image.name.clone(),
        usage: image.usage.map(Into::into),
        width: image.width,
        height: image.height,
        depth: image.depth,
        view_dimension: image.view_dimension.into(),
        image_format: image.image_format.into(),
        mipmap_count: image.mipmap_count,
        image_data: image.image_data.clone(),
    }
}

fn model_root_rs(py: Python, root: &ModelRoot) -> PyResult<xc3_model::ModelRoot> {
    Ok(xc3_model::ModelRoot {
        models: models_rs(py, &root.models.extract(py)?)?,
        buffers: model_buffers_rs(py, &root.buffers.extract(py)?)?,
        image_textures: root
            .image_textures
            .extract::<'_, '_, Vec<ImageTexture>>(py)?
            .iter()
            .map(image_texture_rs)
            .collect(),
        skeleton: root
            .skeleton
            .as_ref()
            .map(|s| skeleton_rs(py, s))
            .transpose()?,
    })
}

fn skeleton_rs(py: Python, skeleton: &Skeleton) -> PyResult<xc3_model::Skeleton> {
    Ok(xc3_model::Skeleton {
        bones: skeleton
            .bones
            .extract::<'_, '_, Vec<Bone>>(py)?
            .iter()
            .map(|b| bone_rs(py, b))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn bone_rs(py: Python, bone: &Bone) -> PyResult<xc3_model::Bone> {
    Ok(xc3_model::Bone {
        name: bone.name.clone(),
        transform: pyarray_to_mat4(py, &bone.transform)?,
        parent_index: bone.parent_index,
    })
}

fn bone_py(bone: xc3_model::Bone, py: Python<'_>) -> Bone {
    Bone {
        name: bone.name,
        transform: mat4_to_pyarray(py, bone.transform),
        parent_index: bone.parent_index,
    }
}

fn model_py(py: Python, model: xc3_model::Model) -> Model {
    Model {
        meshes: PyList::new_bound(
            py,
            model.meshes.into_iter().map(|mesh| {
                Mesh {
                    vertex_buffer_index: mesh.vertex_buffer_index,
                    index_buffer_index: mesh.index_buffer_index,
                    index_buffer_index2: mesh.index_buffer_index2,
                    material_index: mesh.material_index,
                    ext_mesh_index: mesh.ext_mesh_index,
                    lod_item_index: mesh.lod_item_index,
                    flags1: mesh.flags1,
                    flags2: mesh.flags2.into(),
                    base_mesh_index: mesh.base_mesh_index,
                }
                .into_py(py)
            }),
        )
        .into(),
        instances: transforms_pyarray(py, &model.instances),
        model_buffers_index: model.model_buffers_index,
        max_xyz: model.max_xyz.to_array(),
        min_xyz: model.min_xyz.to_array(),
        bounding_radius: model.bounding_radius,
    }
}

fn materials_py(py: Python, materials: Vec<xc3_model::Material>) -> Py<PyList> {
    // TODO: avoid unwrap.
    PyList::new_bound(
        py,
        materials.into_iter().map(|material| {
            Material {
                name: material.name,
                state_flags: material.state_flags.map_py(py).unwrap(),
                textures: PyList::new_bound(
                    py,
                    material.textures.into_iter().map(|texture| {
                        Texture {
                            image_texture_index: texture.image_texture_index,
                            sampler_index: texture.sampler_index,
                        }
                        .into_py(py)
                    }),
                )
                .into(),
                alpha_test: material.alpha_test.map(|a| TextureAlphaTest {
                    texture_index: a.texture_index,
                    channel_index: a.channel_index,
                }),
                shader: material.shader.map(Shader),
                pass_type: material.pass_type.into(),
                parameters: MaterialParameters {
                    mat_color: material.parameters.mat_color,
                    alpha_test_ref: material.parameters.alpha_test_ref,
                    tex_matrix: material.parameters.tex_matrix,
                    work_float4: material.parameters.work_float4,
                    work_color: material.parameters.work_color,
                },
                work_values: material.work_values,
                shader_vars: material.shader_vars,
                work_callbacks: material.work_callbacks,
                alpha_test_ref: material.alpha_test_ref,
                m_unks1_1: material.m_unks1_1,
                m_unks1_2: material.m_unks1_2,
                m_unks1_3: material.m_unks1_3,
                m_unks1_4: material.m_unks1_4,
                technique_index: material.technique_index,
                m_unks2_2: material.m_unks2_2,
                m_unks3_1: material.m_unks3_1,
            }
            .into_py(py)
        }),
    )
    .into()
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

#[pymodule]
fn xc3_model_py(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    // Match the module hierarchy and types of xc3_model as closely as possible.
    animation::animation(py, m)?;
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

    m.add_class::<Shader>()?;
    m.add_class::<Texture>()?;

    m.add_class::<ImageTexture>()?;
    m.add_class::<TextureUsage>()?;
    m.add_class::<ViewDimension>()?;
    m.add_class::<ImageFormat>()?;

    m.add_class::<Sampler>()?;
    m.add_class::<AddressMode>()?;
    m.add_class::<FilterMode>()?;

    m.add_class::<OutputAssignments>()?;
    m.add_class::<OutputAssignment>()?;
    m.add_class::<ChannelAssignment>()?;
    m.add_class::<ChannelAssignmentTexture>()?;

    m.add_class::<Mxmd>()?;
    m.add_class::<Msrd>()?;

    m.add("Xc3ModelError", py.get_type_bound::<Xc3ModelError>())?;

    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(load_model_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(load_map, m)?)?;
    m.add_function(wrap_pyfunction!(load_animations, m)?)?;

    Ok(())
}
