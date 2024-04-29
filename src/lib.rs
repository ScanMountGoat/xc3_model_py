use std::path::Path;

use glam::{Mat4, Vec2, Vec3, Vec4};
use numpy::{IntoPyArray, PyArray};
use pyo3::{create_exception, exceptions::PyException, prelude::*, types::PyList};
use rayon::prelude::*;
use xc3_model::animation::BoneIndex;

// Create a Python class for every public xc3_model type.
// We don't define these conversions on the xc3_model types themselves.
// This flexibility allows more efficient and idiomatic bindings.
// Py<PyList> creates a pure Python list for allowing mutating of elements.
// PyObject supports numpy.ndarray for performance.
// Vec<u8> uses Python's bytes object for performance.

create_exception!(xc3_model_py, Xc3ModelError, PyException);

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
    };
}

// TODO: traits and functions to simplify list conversions?

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct ModelRoot {
    pub models: Models,
    pub buffers: ModelBuffers,
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
pub struct ModelBuffers {
    pub vertex_buffers: Py<PyList>,
    pub index_buffers: Py<PyList>,
    pub weights: Option<Weights>,
}

#[pymethods]
impl ModelBuffers {
    #[new]
    pub fn new(
        vertex_buffers: Py<PyList>,
        index_buffers: Py<PyList>,
        weights: Option<Weights>,
    ) -> Self {
        Self {
            vertex_buffers,
            index_buffers,
            weights,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Weights {
    weight_buffers: Vec<SkinWeights>,
    // TODO: how to handle this?
    weight_groups: xc3_model::skinning::WeightGroups,
}

#[pymethods]
impl Weights {
    #[new]
    pub fn new(weight_buffers: Vec<SkinWeights>) -> Self {
        // TODO: rework xc3_model to make this easier to work with.
        Self {
            weight_buffers,
            weight_groups: xc3_model::skinning::WeightGroups::Groups {
                weight_groups: Vec::new(),
                weight_lods: Vec::new(),
            },
        }
    }

    pub fn weight_buffer(&self, py: Python, flags2: u32) -> PyResult<Option<SkinWeights>> {
        Ok(weights_rs(py, self)?
            .weight_buffer(flags2)
            .map(|b| skin_weights_py(py, b)))
    }

    // TODO: make this a method of WeightGroups?
    pub fn weights_start_index(
        &self,
        skin_flags: u32,
        lod: u16,
        unk_type: RenderPassType,
    ) -> usize {
        self.weight_groups
            .weights_start_index(skin_flags, lod, unk_type.into())
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct SkinWeights {
    // N x 4 numpy.ndarray
    pub bone_indices: PyObject,
    // N x 4 numpy.ndarray
    pub weights: PyObject,
    /// The name list for the indices in [bone_indices](#structfield.bone_indices).
    pub bone_names: Py<PyList>,
}

#[pymethods]
impl SkinWeights {
    #[new]
    pub fn new(bone_indices: PyObject, weights: PyObject, bone_names: Py<PyList>) -> Self {
        Self {
            bone_indices,
            weights,
            bone_names,
        }
    }

    pub fn to_influences(&self, py: Python, weight_indices: PyObject) -> PyResult<Vec<Influence>> {
        let weight_indices: Vec<_> = weight_indices.extract(py)?;
        let influences = skin_weights_rs(py, self)?.to_influences(&weight_indices);
        Ok(influences_py(py, influences))
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct Models {
    pub models: Py<PyList>,
    pub materials: Py<PyList>,
    pub samplers: Py<PyList>,
    pub base_lod_indices: Option<Vec<u16>>,
    pub morph_controller_names: Py<PyList>,
    pub animation_morph_names: Py<PyList>,
    pub max_xyz: [f32; 3],
    pub min_xyz: [f32; 3],
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
        base_lod_indices: Option<Vec<u16>>,
    ) -> Self {
        Self {
            models,
            materials,
            samplers,
            base_lod_indices,
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
    pub material_index: usize,
    pub lod: u16,
    pub flags1: u32,
    pub flags2: u32,
}

#[pymethods]
impl Mesh {
    #[new]
    pub fn new(
        vertex_buffer_index: usize,
        index_buffer_index: usize,
        material_index: usize,
        lod: u16,
        flags1: u32,
        flags2: u32,
    ) -> Self {
        Self {
            vertex_buffer_index,
            index_buffer_index,
            material_index,
            lod,
            flags1,
            flags2,
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
    // pub flags: StateFlags,
    pub textures: Py<PyList>,
    pub alpha_test: Option<TextureAlphaTest>,
    pub shader: Option<Shader>,
    pub pass_type: RenderPassType,
    pub parameters: MaterialParameters,
}

#[pymethods]
impl Material {
    #[new]
    fn new(
        name: String,
        textures: Py<PyList>,
        pass_type: RenderPassType,
        parameters: MaterialParameters,
        alpha_test: Option<TextureAlphaTest>,
        shader: Option<Shader>,
    ) -> Self {
        Self {
            name,
            textures,
            alpha_test,
            shader,
            pass_type,
            parameters,
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

        Ok(OutputAssignments {
            assignments: assignments.assignments.map(|a| OutputAssignment {
                x: a.x.map(ChannelAssignment),
                y: a.y.map(ChannelAssignment),
                z: a.z.map(ChannelAssignment),
                w: a.w.map(ChannelAssignment),
            }),
        })
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
#[derive(Debug, Clone)]
pub struct TextureAlphaTest {
    pub texture_index: usize,
    pub channel_index: usize,
    pub ref_value: f32,
}

#[pymethods]
impl TextureAlphaTest {
    #[new]
    fn new(texture_index: usize, channel_index: usize, ref_value: f32) -> Self {
        Self {
            texture_index,
            channel_index,
            ref_value,
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

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct VertexBuffer {
    pub attributes: Py<PyList>,
    pub morph_targets: Py<PyList>,
    pub outline_buffer_index: Option<usize>,
}

#[pymethods]
impl VertexBuffer {
    #[new]
    fn new(
        attributes: Py<PyList>,
        morph_targets: Py<PyList>,
        outline_buffer_index: Option<usize>,
    ) -> Self {
        Self {
            attributes,
            morph_targets,
            outline_buffer_index,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct AttributeData {
    pub attribute_type: AttributeType,
    // numpy.ndarray with vertex count many rows
    pub data: PyObject,
}

#[pymethods]
impl AttributeData {
    #[new]
    fn new(attribute_type: AttributeType, data: PyObject) -> Self {
        Self {
            attribute_type,
            data,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum AttributeType {
    Position,
    Normal,
    Tangent,
    TexCoord0,
    TexCoord1,
    TexCoord2,
    TexCoord3,
    TexCoord4,
    TexCoord5,
    TexCoord6,
    TexCoord7,
    TexCoord8,
    VertexColor,
    Blend,
    WeightIndex,
    SkinWeights,
    BoneIndices,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct MorphTarget {
    pub morph_controller_index: usize,
    // N x 3 numpy.ndarray
    pub position_deltas: PyObject,
    // N x 4 numpy.ndarray
    pub normal_deltas: PyObject,
    // N x 4 numpy.ndarray
    pub tangent_deltas: PyObject,
    pub vertex_indices: PyObject,
}

#[pymethods]
impl MorphTarget {
    #[new]
    fn new(
        morph_controller_index: usize,
        position_deltas: PyObject,
        normal_deltas: PyObject,
        tangent_deltas: PyObject,
        vertex_indices: PyObject,
    ) -> Self {
        Self {
            morph_controller_index,
            position_deltas,
            normal_deltas,
            tangent_deltas,
            vertex_indices,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct Influence {
    pub bone_name: String,
    pub weights: Py<PyList>,
}

#[pymethods]
impl Influence {
    #[new]
    fn new(bone_name: String, weights: Py<PyList>) -> Self {
        Self { bone_name, weights }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct VertexWeight {
    pub vertex_index: u32,
    pub weight: f32,
}

#[pymethods]
impl VertexWeight {
    #[new]
    fn new(vertex_index: u32, weight: f32) -> Self {
        Self {
            vertex_index,
            weight,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct IndexBuffer {
    pub indices: PyObject,
}

#[pymethods]
impl IndexBuffer {
    #[new]
    fn new(indices: PyObject) -> Self {
        Self { indices }
    }
}

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
#[derive(Debug, Clone)]
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
pub struct Animation {
    pub name: String,
    pub space_mode: SpaceMode,
    pub play_mode: PlayMode,
    pub blend_mode: BlendMode,
    pub frames_per_second: f32,
    pub frame_count: u32,
    pub tracks: Vec<Track>,
}

#[pymethods]
impl Animation {
    pub fn current_frame(&self, current_time_seconds: f32) -> f32 {
        // TODO: looping?
        current_time_seconds * self.frames_per_second
    }

    pub fn skinning_transforms(
        &self,
        py: Python,
        skeleton: Skeleton,
        frame: f32,
    ) -> PyResult<PyObject> {
        let animation = animation_rs(self);
        let skeleton = skeleton_rs(py, &skeleton)?;
        let transforms = animation.skinning_transforms(&skeleton, frame);
        Ok(transforms_pyarray(py, &transforms))
    }

    pub fn model_space_transforms(
        &self,
        py: Python,
        skeleton: Skeleton,
        frame: f32,
    ) -> PyResult<PyObject> {
        let animation = animation_rs(self);
        let skeleton = skeleton_rs(py, &skeleton)?;
        let transforms = animation.model_space_transforms(&skeleton, frame);
        Ok(transforms_pyarray(py, &transforms))
    }

    pub fn local_space_transforms(
        &self,
        py: Python,
        skeleton: Skeleton,
        frame: f32,
    ) -> PyResult<PyObject> {
        let animation = animation_rs(self);
        let skeleton = skeleton_rs(py, &skeleton)?;
        let transforms = animation.local_space_transforms(&skeleton, frame);
        Ok(transforms_pyarray(py, &transforms))
    }
}

// TODO: Expose implementation details?
#[pyclass]
#[derive(Debug, Clone)]
pub struct Track(xc3_model::animation::Track);

#[pymethods]
impl Track {
    pub fn sample_translation(&self, frame: f32) -> Option<(f32, f32, f32)> {
        self.0.sample_translation(frame).map(Into::into)
    }

    pub fn sample_rotation(&self, frame: f32) -> Option<(f32, f32, f32, f32)> {
        self.0.sample_rotation(frame).map(Into::into)
    }

    pub fn sample_scale(&self, frame: f32) -> Option<(f32, f32, f32)> {
        self.0.sample_scale(frame).map(Into::into)
    }

    pub fn sample_transform(&self, py: Python, frame: f32) -> Option<PyObject> {
        self.0
            .sample_transform(frame)
            .map(|t| mat4_to_pyarray(py, t))
    }

    // Workaround for representing Rust enums in Python.
    pub fn bone_index(&self) -> Option<usize> {
        match &self.0.bone_index {
            BoneIndex::Index(index) => Some(*index),
            _ => None,
        }
    }

    pub fn bone_hash(&self) -> Option<u32> {
        match &self.0.bone_index {
            BoneIndex::Hash(hash) => Some(*hash),
            _ => None,
        }
    }

    pub fn bone_name(&self) -> Option<&str> {
        match &self.0.bone_index {
            BoneIndex::Name(name) => Some(name),
            _ => None,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub x_coeffs: (f32, f32, f32, f32),
    pub y_coeffs: (f32, f32, f32, f32),
    pub z_coeffs: (f32, f32, f32, f32),
    pub w_coeffs: (f32, f32, f32, f32),
}

python_enum!(SpaceMode, xc3_model::animation::SpaceMode, Local, Model);
python_enum!(PlayMode, xc3_model::animation::PlayMode, Loop, Single);
python_enum!(BlendMode, xc3_model::animation::BlendMode, Blend, Add);

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
        models: Models,
        buffers: ModelBuffers,
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
            .extract::<'_, Vec<ImageTexture>>(py)?
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
            .map(|buffer| buffer.into_pyarray(py).into())
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
        self.image_textures
            .extract::<'_, Vec<ImageTexture>>(py)?
            .par_iter()
            .enumerate()
            .map(|(i, texture)| {
                // TODO: Better way to handle missing name?
                let filename = texture
                    .name
                    .as_ref()
                    .map(|n| format!("{prefix}.{n}.{ext}"))
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
            .extract::<'_, Vec<ImageTexture>>(py)?
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
            .map(|buffer| buffer.into_pyarray(py).into())
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
        self.image_textures
            .extract::<'_, Vec<ImageTexture>>(py)?
            .par_iter()
            .enumerate()
            .map(|(i, texture)| {
                // TODO: Better way to handle missing name?
                let filename = texture
                    .name
                    .as_ref()
                    .map(|n| format!("{prefix}.{n}.{ext}"))
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
    // TODO: support texture edits as well?
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
            xc3_model::ChannelAssignment::Value(_) => None,
        }
    }

    pub fn value(&self) -> Option<f32> {
        match self.0 {
            xc3_model::ChannelAssignment::Texture { .. } => None,
            xc3_model::ChannelAssignment::Value(f) => Some(f),
        }
    }
}

fn influences_py(py: Python, influences: Vec<xc3_model::skinning::Influence>) -> Vec<Influence> {
    influences
        .into_iter()
        .map(|i| Influence {
            bone_name: i.bone_name,
            weights: PyList::new(
                py,
                i.weights.into_iter().map(|w| {
                    VertexWeight {
                        vertex_index: w.vertex_index,
                        weight: w.weight,
                    }
                    .into_py(py)
                }),
            )
            .into(),
        })
        .collect()
}

#[pyfunction]
fn load_model(py: Python, wimdo_path: &str, database_path: Option<&str>) -> PyResult<ModelRoot> {
    let database = database_path
        .map(xc3_model::shader_database::ShaderDatabase::from_file)
        .transpose()
        .map_err(py_exception)?;
    let root = xc3_model::load_model(wimdo_path, database.as_ref()).map_err(py_exception)?;
    Ok(model_root_py(py, root))
}

#[pyfunction]
fn load_model_legacy(py: Python, camdo_path: &str) -> PyResult<ModelRoot> {
    let root = xc3_model::load_model_legacy(camdo_path);
    Ok(model_root_py(py, root))
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
    Ok(roots
        .into_iter()
        .map(|root| map_root_py(py, root))
        .collect())
}

#[pyfunction]
fn load_animations(_py: Python, anim_path: &str) -> PyResult<Vec<Animation>> {
    let animations = xc3_model::load_animations(anim_path).map_err(py_exception)?;
    Ok(animations.into_iter().map(animation_py).collect())
}

#[pyfunction]
fn murmur3(name: &str) -> u32 {
    xc3_model::animation::murmur3(name.as_bytes())
}

fn py_exception<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<Xc3ModelError, _>(format!("{e}"))
}

fn map_root_py(py: Python, root: xc3_model::MapRoot) -> MapRoot {
    MapRoot {
        groups: PyList::new(
            py,
            root.groups
                .into_iter()
                .map(|group| model_group_py(py, group).into_py(py)),
        )
        .into(),
        image_textures: PyList::new(
            py,
            root.image_textures
                .into_iter()
                .map(|t| image_texture_py(t).into_py(py)),
        )
        .into(),
    }
}

fn model_root_py(py: Python, root: xc3_model::ModelRoot) -> ModelRoot {
    ModelRoot {
        models: models_py(py, root.models),
        buffers: model_buffers_py(py, root.buffers),
        image_textures: PyList::new(
            py,
            root.image_textures
                .into_iter()
                .map(|t| image_texture_py(t).into_py(py)),
        )
        .into(),
        skeleton: root.skeleton.map(|skeleton| Skeleton {
            bones: PyList::new(
                py,
                skeleton
                    .bones
                    .into_iter()
                    .map(|bone| bone_py(bone, py).into_py(py)),
            )
            .into(),
        }),
    }
}

fn models_py(py: Python, models: xc3_model::Models) -> Models {
    Models {
        models: PyList::new(
            py,
            models
                .models
                .into_iter()
                .map(|m| model_py(py, m).into_py(py)),
        )
        .into(),
        materials: materials_py(py, models.materials),
        samplers: PyList::new(
            py,
            models.samplers.iter().map(|s| sampler_py(s).into_py(py)),
        )
        .into(),
        base_lod_indices: models.base_lod_indices,
        morph_controller_names: PyList::new(py, models.morph_controller_names).into(),
        animation_morph_names: PyList::new(py, models.animation_morph_names).into(),
        max_xyz: models.max_xyz.to_array(),
        min_xyz: models.min_xyz.to_array(),
    }
}

fn models_rs(py: Python, models: &Models) -> PyResult<xc3_model::Models> {
    Ok(xc3_model::Models {
        models: models
            .models
            .extract::<'_, Vec<Model>>(py)?
            .iter()
            .map(|model| model_rs(py, model))
            .collect::<PyResult<Vec<_>>>()?,
        materials: models
            .materials
            .extract::<'_, Vec<Material>>(py)?
            .iter()
            .map(|m| material_rs(py, m))
            .collect::<PyResult<Vec<_>>>()?,
        samplers: models
            .samplers
            .extract::<'_, Vec<Sampler>>(py)?
            .iter()
            .map(sampler_rs)
            .collect(),
        base_lod_indices: models.base_lod_indices.clone(),
        morph_controller_names: models.morph_controller_names.extract(py)?,
        animation_morph_names: models.animation_morph_names.extract(py)?,
        max_xyz: models.max_xyz.into(),
        min_xyz: models.min_xyz.into(),
    })
}

fn model_group_py(py: Python, group: xc3_model::ModelGroup) -> ModelGroup {
    ModelGroup {
        models: PyList::new(
            py,
            group
                .models
                .into_iter()
                .map(|models| models_py(py, models).into_py(py)),
        )
        .into(),
        buffers: PyList::new(
            py,
            group
                .buffers
                .into_iter()
                .map(|buffer| model_buffers_py(py, buffer).into_py(py)),
        )
        .into(),
    }
}

fn model_rs(py: Python, model: &Model) -> PyResult<xc3_model::Model> {
    Ok(xc3_model::Model {
        meshes: model
            .meshes
            .extract::<'_, Vec<Mesh>>(py)?
            .iter()
            .map(|mesh| xc3_model::Mesh {
                vertex_buffer_index: mesh.vertex_buffer_index,
                index_buffer_index: mesh.index_buffer_index,
                material_index: mesh.material_index,
                lod: mesh.lod,
                flags1: mesh.flags1,
                flags2: mesh.flags2.try_into().unwrap(),
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
    Ok(xc3_model::Material {
        name: material.name.clone(),
        // TODO: proper state flags and defaults?
        flags: xc3_model::StateFlags {
            depth_write_mode: 0,
            blend_mode: xc3_model::BlendMode::Disabled,
            cull_mode: xc3_model::CullMode::Disabled,
            unk4: 0,
            stencil_value: xc3_model::StencilValue::Unk0,
            stencil_mode: xc3_model::StencilMode::Unk0,
            depth_func: xc3_model::DepthFunc::Equal,
            color_write_mode: 0,
        },
        textures: material
            .textures
            .extract::<'_, Vec<Texture>>(py)?
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
                ref_value: a.ref_value,
            }),
        shader: material.shader.clone().map(|s| s.0),
        pass_type: xc3_model::RenderPassType::Unk0,
        parameters: xc3_model::MaterialParameters {
            mat_color: material.parameters.mat_color,
            alpha_test_ref: material.parameters.alpha_test_ref,
            tex_matrix: material.parameters.tex_matrix.clone(),
            work_float4: material.parameters.work_float4.clone(),
            work_color: material.parameters.work_color.clone(),
        },
    })
}

fn model_buffers_rs(
    py: Python,
    buffer: &ModelBuffers,
) -> PyResult<xc3_model::vertex::ModelBuffers> {
    Ok(xc3_model::vertex::ModelBuffers {
        vertex_buffers: buffer
            .vertex_buffers
            .extract::<'_, Vec<VertexBuffer>>(py)?
            .iter()
            .map(|b| vertex_buffer_rs(py, b))
            .collect::<PyResult<Vec<_>>>()?,
        // TODO: Fill in all fields
        outline_buffers: Vec::new(),
        index_buffers: buffer
            .index_buffers
            .extract::<'_, Vec<IndexBuffer>>(py)?
            .iter()
            .map(|b| {
                Ok(xc3_model::vertex::IndexBuffer {
                    indices: b.indices.extract(py)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        unk_buffers: Vec::new(),
        weights: buffer
            .weights
            .as_ref()
            .map(|w| weights_rs(py, w))
            .transpose()?,
    })
}

fn model_buffers_py(py: Python, buffer: xc3_model::vertex::ModelBuffers) -> ModelBuffers {
    ModelBuffers {
        vertex_buffers: vertex_buffers_py(py, buffer.vertex_buffers),
        index_buffers: index_buffers_py(py, buffer.index_buffers),
        weights: buffer.weights.map(|weights| Weights {
            weight_buffers: weight_buffers_py(py, weights.weight_buffers),
            weight_groups: weights.weight_groups,
        }),
    }
}

fn weights_rs(py: Python, w: &Weights) -> PyResult<xc3_model::skinning::Weights> {
    Ok(xc3_model::skinning::Weights {
        weight_buffers: weight_buffers_rs(py, &w.weight_buffers)?,
        weight_groups: w.weight_groups.clone(),
    })
}

fn vertex_buffer_rs(py: Python, b: &VertexBuffer) -> PyResult<xc3_model::vertex::VertexBuffer> {
    Ok(xc3_model::vertex::VertexBuffer {
        attributes: b
            .attributes
            .extract::<'_, Vec<AttributeData>>(py)?
            .iter()
            .map(|a| attribute_data_rs(py, a))
            .collect::<PyResult<Vec<_>>>()?,
        morph_targets: b
            .morph_targets
            .extract::<'_, Vec<MorphTarget>>(py)?
            .iter()
            .map(|t| {
                Ok(xc3_model::vertex::MorphTarget {
                    morph_controller_index: t.morph_controller_index,
                    position_deltas: pyarray_to_vec3s(py, &t.position_deltas)?,
                    normal_deltas: pyarray_to_vec4s(py, &t.normal_deltas)?,
                    tangent_deltas: pyarray_to_vec4s(py, &t.tangent_deltas)?,
                    vertex_indices: t.vertex_indices.extract(py)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        outline_buffer_index: b.outline_buffer_index,
    })
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
        models: models_rs(py, &root.models)?,
        buffers: model_buffers_rs(py, &root.buffers)?,
        image_textures: root
            .image_textures
            .extract::<'_, Vec<ImageTexture>>(py)?
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

fn weight_buffers_rs(
    py: Python,
    weight_buffers: &[SkinWeights],
) -> Result<Vec<xc3_model::skinning::SkinWeights>, PyErr> {
    weight_buffers
        .iter()
        .map(|w| skin_weights_rs(py, w))
        .collect()
}

fn skin_weights_rs(py: Python, w: &SkinWeights) -> PyResult<xc3_model::skinning::SkinWeights> {
    Ok(xc3_model::skinning::SkinWeights {
        bone_indices: w.bone_indices.extract(py)?,
        weights: pyarray_to_vec4s(py, &w.weights)?,
        bone_names: w.bone_names.extract(py)?,
    })
}

fn skin_weights_py(py: Python, w: xc3_model::skinning::SkinWeights) -> SkinWeights {
    SkinWeights {
        bone_indices: uvec4_pyarray(py, &w.bone_indices),
        weights: vec4s_pyarray(py, &w.weights),
        bone_names: PyList::new(py, &w.bone_names).into(),
    }
}

fn weight_buffers_py(
    py: Python,
    weight_buffers: Vec<xc3_model::skinning::SkinWeights>,
) -> Vec<SkinWeights> {
    weight_buffers
        .into_iter()
        .map(|w| skin_weights_py(py, w))
        .collect()
}

fn skeleton_rs(py: Python, skeleton: &Skeleton) -> PyResult<xc3_model::Skeleton> {
    Ok(xc3_model::Skeleton {
        bones: skeleton
            .bones
            .extract::<'_, Vec<Bone>>(py)?
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
        meshes: PyList::new(
            py,
            model.meshes.into_iter().map(|mesh| {
                Mesh {
                    vertex_buffer_index: mesh.vertex_buffer_index,
                    index_buffer_index: mesh.index_buffer_index,
                    material_index: mesh.material_index,
                    lod: mesh.lod,
                    flags1: mesh.flags1,
                    flags2: mesh.flags2.into(),
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
    PyList::new(
        py,
        materials.into_iter().map(|material| {
            Material {
                name: material.name,
                textures: PyList::new(
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
                    ref_value: a.ref_value,
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
            }
            .into_py(py)
        }),
    )
    .into()
}

fn sampler_py(sampler: &xc3_model::Sampler) -> Sampler {
    Sampler {
        address_mode_u: sampler.address_mode_u.into(),
        address_mode_v: sampler.address_mode_v.into(),
        address_mode_w: sampler.address_mode_w.into(),
        min_filter: sampler.min_filter.into(),
        mag_filter: sampler.mag_filter.into(),
        mip_filter: sampler.mip_filter.into(),
        mipmaps: sampler.mipmaps,
    }
}

fn sampler_rs(s: &Sampler) -> xc3_model::Sampler {
    xc3_model::Sampler {
        address_mode_u: s.address_mode_u.into(),
        address_mode_v: s.address_mode_v.into(),
        address_mode_w: s.address_mode_w.into(),
        min_filter: s.min_filter.into(),
        mag_filter: s.mag_filter.into(),
        mip_filter: s.mip_filter.into(),
        mipmaps: s.mipmaps,
    }
}

fn vertex_buffers_py(
    py: Python,
    vertex_buffers: Vec<xc3_model::vertex::VertexBuffer>,
) -> Py<PyList> {
    PyList::new(
        py,
        vertex_buffers.into_iter().map(|buffer| {
            VertexBuffer {
                attributes: vertex_attributes_py(py, buffer.attributes),
                morph_targets: morph_targets_py(py, buffer.morph_targets),
                outline_buffer_index: buffer.outline_buffer_index,
            }
            .into_py(py)
        }),
    )
    .into()
}

fn vertex_attributes_py(
    py: Python,
    attributes: Vec<xc3_model::vertex::AttributeData>,
) -> Py<PyList> {
    PyList::new(
        py,
        attributes
            .into_iter()
            .map(|attribute| attribute_data_py(py, attribute).into_py(py)),
    )
    .into()
}

fn attribute_data_py(py: Python, attribute: xc3_model::vertex::AttributeData) -> AttributeData {
    match attribute {
        xc3_model::vertex::AttributeData::Position(values) => AttributeData {
            attribute_type: AttributeType::Position,
            data: vec3s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::Normal(values) => AttributeData {
            attribute_type: AttributeType::Normal,
            data: vec4s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::Tangent(values) => AttributeData {
            attribute_type: AttributeType::Tangent,
            data: vec4s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord0(values) => AttributeData {
            attribute_type: AttributeType::TexCoord0,
            data: vec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord1(values) => AttributeData {
            attribute_type: AttributeType::TexCoord1,
            data: vec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord2(values) => AttributeData {
            attribute_type: AttributeType::TexCoord2,
            data: vec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord3(values) => AttributeData {
            attribute_type: AttributeType::TexCoord3,
            data: vec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord4(values) => AttributeData {
            attribute_type: AttributeType::TexCoord4,
            data: vec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord5(values) => AttributeData {
            attribute_type: AttributeType::TexCoord5,
            data: vec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord6(values) => AttributeData {
            attribute_type: AttributeType::TexCoord6,
            data: vec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord7(values) => AttributeData {
            attribute_type: AttributeType::TexCoord7,
            data: vec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord8(values) => AttributeData {
            attribute_type: AttributeType::TexCoord8,
            data: vec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::VertexColor(values) => AttributeData {
            attribute_type: AttributeType::VertexColor,
            data: vec4s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::Blend(values) => AttributeData {
            attribute_type: AttributeType::Blend,
            data: vec4s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::WeightIndex(values) => AttributeData {
            attribute_type: AttributeType::WeightIndex,
            data: uvec2s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::SkinWeights(values) => AttributeData {
            attribute_type: AttributeType::SkinWeights,
            data: vec4s_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::BoneIndices(values) => AttributeData {
            attribute_type: AttributeType::BoneIndices,
            data: uvec4_pyarray(py, &values),
        },
    }
}

fn attribute_data_rs(
    py: Python,
    attribute: &AttributeData,
) -> PyResult<xc3_model::vertex::AttributeData> {
    use xc3_model::vertex::AttributeData as AttrRs;

    match attribute.attribute_type {
        AttributeType::Position => Ok(AttrRs::Position(pyarray_to_vec3s(py, &attribute.data)?)),
        AttributeType::Normal => Ok(AttrRs::Normal(pyarray_to_vec4s(py, &attribute.data)?)),
        AttributeType::Tangent => Ok(AttrRs::Tangent(pyarray_to_vec4s(py, &attribute.data)?)),
        AttributeType::TexCoord0 => Ok(AttrRs::TexCoord0(pyarray_to_vec2s(py, &attribute.data)?)),
        AttributeType::TexCoord1 => Ok(AttrRs::TexCoord1(pyarray_to_vec2s(py, &attribute.data)?)),
        AttributeType::TexCoord2 => Ok(AttrRs::TexCoord2(pyarray_to_vec2s(py, &attribute.data)?)),
        AttributeType::TexCoord3 => Ok(AttrRs::TexCoord3(pyarray_to_vec2s(py, &attribute.data)?)),
        AttributeType::TexCoord4 => Ok(AttrRs::TexCoord4(pyarray_to_vec2s(py, &attribute.data)?)),
        AttributeType::TexCoord5 => Ok(AttrRs::TexCoord5(pyarray_to_vec2s(py, &attribute.data)?)),
        AttributeType::TexCoord6 => Ok(AttrRs::TexCoord6(pyarray_to_vec2s(py, &attribute.data)?)),
        AttributeType::TexCoord7 => Ok(AttrRs::TexCoord7(pyarray_to_vec2s(py, &attribute.data)?)),
        AttributeType::TexCoord8 => Ok(AttrRs::TexCoord8(pyarray_to_vec2s(py, &attribute.data)?)),
        AttributeType::VertexColor => {
            Ok(AttrRs::VertexColor(pyarray_to_vec4s(py, &attribute.data)?))
        }
        AttributeType::Blend => Ok(AttrRs::Blend(pyarray_to_vec4s(py, &attribute.data)?)),
        AttributeType::WeightIndex => Ok(AttrRs::WeightIndex(attribute.data.extract(py)?)),
        AttributeType::SkinWeights => {
            Ok(AttrRs::SkinWeights(pyarray_to_vec4s(py, &attribute.data)?))
        }
        AttributeType::BoneIndices => Ok(AttrRs::BoneIndices(attribute.data.extract(py)?)),
    }
}

fn morph_targets_py(py: Python, targets: Vec<xc3_model::vertex::MorphTarget>) -> Py<PyList> {
    PyList::new(
        py,
        targets.into_iter().map(|target| {
            MorphTarget {
                morph_controller_index: target.morph_controller_index,
                position_deltas: vec3s_pyarray(py, &target.position_deltas),
                normal_deltas: vec4s_pyarray(py, &target.normal_deltas),
                tangent_deltas: vec4s_pyarray(py, &target.tangent_deltas),
                vertex_indices: target.vertex_indices.into_pyarray(py).into(),
            }
            .into_py(py)
        }),
    )
    .into()
}

fn index_buffers_py(py: Python, index_buffers: Vec<xc3_model::vertex::IndexBuffer>) -> Py<PyList> {
    PyList::new(
        py,
        index_buffers.into_iter().map(|buffer| {
            IndexBuffer {
                indices: buffer.indices.into_pyarray(py).into(),
            }
            .into_py(py)
        }),
    )
    .into()
}

fn animation_rs(animation: &Animation) -> xc3_model::animation::Animation {
    xc3_model::animation::Animation {
        name: animation.name.clone(),
        space_mode: animation.space_mode.into(),
        play_mode: animation.play_mode.into(),
        blend_mode: animation.blend_mode.into(),
        frames_per_second: animation.frames_per_second,
        frame_count: animation.frame_count,
        tracks: animation.tracks.iter().map(|t| t.0.clone()).collect(),
        morph_tracks: None, // TODO: morph animations?
    }
}

fn animation_py(animation: xc3_model::animation::Animation) -> Animation {
    Animation {
        name: animation.name.clone(),
        space_mode: animation.space_mode.into(),
        play_mode: animation.play_mode.into(),
        blend_mode: animation.blend_mode.into(),
        frames_per_second: animation.frames_per_second,
        frame_count: animation.frame_count,
        tracks: animation.tracks.into_iter().map(Track).collect(),
    }
}

// TODO: Share code?
fn vec2s_pyarray(py: Python, values: &[Vec2]) -> PyObject {
    // This flatten will be optimized in Release mode.
    // This avoids needing unsafe code.
    let count = values.len();
    values
        .iter()
        .flat_map(|v| v.to_array())
        .collect::<Vec<f32>>()
        .into_pyarray(py)
        .reshape((count, 2))
        .unwrap()
        .into()
}

fn pyarray_to_vec2s(py: Python, values: &PyObject) -> PyResult<Vec<Vec2>> {
    let values: Vec<[f32; 2]> = values.extract(py)?;
    Ok(values.into_iter().map(Into::into).collect())
}

fn vec3s_pyarray(py: Python, values: &[Vec3]) -> PyObject {
    // This flatten will be optimized in Release mode.
    // This avoids needing unsafe code.
    let count = values.len();
    values
        .iter()
        .flat_map(|v| v.to_array())
        .collect::<Vec<f32>>()
        .into_pyarray(py)
        .reshape((count, 3))
        .unwrap()
        .into()
}

fn pyarray_to_vec3s(py: Python, values: &PyObject) -> PyResult<Vec<Vec3>> {
    let values: Vec<[f32; 3]> = values.extract(py)?;
    Ok(values.into_iter().map(Into::into).collect())
}

fn vec4s_pyarray(py: Python, values: &[Vec4]) -> PyObject {
    // This flatten will be optimized in Release mode.
    // This avoids needing unsafe code.
    let count = values.len();
    values
        .iter()
        .flat_map(|v| v.to_array())
        .collect::<Vec<f32>>()
        .into_pyarray(py)
        .reshape((count, 4))
        .unwrap()
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
        .into_pyarray(py)
        .reshape((count, 2))
        .unwrap()
        .into()
}

fn pyarray_to_vec4s(py: Python, values: &PyObject) -> PyResult<Vec<Vec4>> {
    let values: Vec<[f32; 4]> = values.extract(py)?;
    Ok(values.into_iter().map(Into::into).collect())
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
        .into_pyarray(py)
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
        .into_pyarray(py)
        .reshape((transform_count, 4, 4))
        .unwrap()
        .into()
}

// TODO: test cases for conversions.
fn mat4_to_pyarray(py: Python, transform: Mat4) -> PyObject {
    PyArray::from_slice(py, &transform.to_cols_array())
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
fn xc3_model_py(py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    // Match the module hierarchy and types of xc3_model as closely as possible.
    animation(py, m)?;
    skinning(py, m)?;
    vertex(py, m)?;

    m.add_class::<ModelRoot>()?;
    m.add_class::<MapRoot>()?;
    m.add_class::<ModelGroup>()?;

    m.add_class::<Models>()?;
    m.add_class::<Model>()?;
    m.add_class::<Mesh>()?;

    m.add_class::<Skeleton>()?;
    m.add_class::<Bone>()?;

    m.add_class::<Material>()?;
    m.add_class::<TextureAlphaTest>()?;
    m.add_class::<MaterialParameters>()?;
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

    m.add("Xc3ModelError", py.get_type::<Xc3ModelError>())?;

    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(load_model_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(load_map, m)?)?;
    m.add_function(wrap_pyfunction!(load_animations, m)?)?;

    Ok(())
}

fn animation(py: Python, module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "animation")?;

    m.add_class::<Animation>()?;
    m.add_class::<Track>()?;
    m.add_class::<Keyframe>()?;
    m.add_class::<SpaceMode>()?;
    m.add_class::<PlayMode>()?;
    m.add_class::<BlendMode>()?;
    m.add_function(wrap_pyfunction!(murmur3, m)?)?;

    module.add_submodule(m)?;
    Ok(())
}

fn vertex(py: Python, module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "vertex")?;

    m.add_class::<ModelBuffers>()?;
    m.add_class::<VertexBuffer>()?;
    m.add_class::<IndexBuffer>()?;
    m.add_class::<AttributeData>()?;
    m.add_class::<AttributeType>()?;
    m.add_class::<MorphTarget>()?;

    module.add_submodule(m)?;
    Ok(())
}

fn skinning(py: Python, module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "skinning")?;

    m.add_class::<Weights>()?;
    m.add_class::<SkinWeights>()?;
    m.add_class::<Influence>()?;
    m.add_class::<VertexWeight>()?;

    module.add_submodule(m)?;
    Ok(())
}
