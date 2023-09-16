use glam::{Mat4, Vec2, Vec3, Vec4};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

// Create a Python class for every public xc3_model type.
// We don't define these conversions on the xc3_model types themselves.
// This flexibility allows more efficient and idiomatic bindings.
#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct ModelRoot {
    pub groups: Vec<ModelGroup>,
    pub image_textures: Vec<ImageTexture>,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct ModelGroup {
    pub models: Vec<Models>,
    pub buffers: Vec<ModelBuffers>,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct ModelBuffers {
    pub vertex_buffers: Vec<VertexBuffer>,
    pub index_buffers: Vec<IndexBuffer>,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Models {
    pub models: Vec<Model>,
    pub materials: Vec<Material>,
    pub samplers: Vec<Sampler>,
    pub skeleton: Option<Skeleton>,
    pub base_lod_indices: Option<Vec<u16>>,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    // N x 4 x 4 numpy.ndarray
    pub instances: PyObject,
    pub model_buffers_index: usize,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertex_buffer_index: usize,
    pub index_buffer_index: usize,
    pub material_index: usize,
    pub lod: u16,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Bone {
    pub name: String,
    pub transform: PyObject,
    pub parent_index: Option<usize>,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    // TODO: how to handle flags?
    // pub flags: StateFlags,
    pub textures: Vec<Texture>,
    pub alpha_test: Option<TextureAlphaTest>,
    pub shader: Option<Shader>,
    // pub unk_type: ShaderUnkType,
    pub parameters: MaterialParameters,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct TextureAlphaTest {
    pub texture_index: usize,
    pub channel_index: usize,
    pub ref_value: f32,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct MaterialParameters {
    pub mat_color: (f32, f32, f32, f32),
    pub alpha_test_ref: f32,
    pub tex_matrix: Option<Vec<[f32; 8]>>,
    pub work_float4: Option<Vec<(f32, f32, f32, f32)>>,
    pub work_color: Option<Vec<(f32, f32, f32, f32)>>,
}

// TODO: Expose implementation details?
#[pyclass]
#[derive(Debug, Clone)]
pub struct Shader(xc3_model::Shader);

#[pymethods]
impl Shader {
    pub fn sampler_channel_index(&self, output_index: usize, channel: char) -> Option<(u32, u32)> {
        self.0.sampler_channel_index(output_index, channel)
    }

    pub fn float_constant(&self, output_index: usize, channel: char) -> Option<f32> {
        self.0.float_constant(output_index, channel)
    }

    pub fn buffer_parameter(&self, output_index: usize, channel: char) -> Option<BufferParameter> {
        self.0
            .buffer_parameter(output_index, channel)
            .map(|b| BufferParameter {
                buffer: b.buffer,
                uniform: b.uniform,
                index: b.index,
                channel: b.channel,
            })
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct BufferParameter {
    pub buffer: String,
    pub uniform: String,
    pub index: usize,
    pub channel: char,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Texture {
    pub image_texture_index: usize,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct VertexBuffer {
    // TODO: Types for this?
    pub attributes: Vec<AttributeData>,
    pub influences: Vec<Influence>,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct AttributeData {
    pub attribute_type: AttributeType,
    // numpy.ndarray with vertex count many rows
    pub data: PyObject,
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum AttributeType {
    Position,
    Normal,
    Tangent,
    Uv1,
    Uv2,
    VertexColor,
    WeightIndex,
    SkinWeights,
    BoneIndices,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Influence {
    pub bone_name: String,
    pub weights: Vec<SkinWeight>,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct SkinWeight {
    pub vertex_index: u32,
    pub weight: f32,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct IndexBuffer {
    pub indices: PyObject,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct ImageTexture {
    pub name: Option<String>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub view_dimension: ViewDimension,
    pub image_format: ImageFormat,
    pub mipmap_count: u32,
    pub image_data: Vec<u8>,
}

// TODO: Create a macro for this?
#[pyclass]
#[derive(Debug, Clone, Copy)]
pub enum ViewDimension {
    D2,
    D3,
    Cube,
}

impl From<xc3_model::ViewDimension> for ViewDimension {
    fn from(value: xc3_model::ViewDimension) -> Self {
        match value {
            xc3_model::ViewDimension::D2 => Self::D2,
            xc3_model::ViewDimension::D3 => Self::D3,
            xc3_model::ViewDimension::Cube => Self::Cube,
        }
    }
}

impl From<ViewDimension> for xc3_model::ViewDimension {
    fn from(value: ViewDimension) -> Self {
        match value {
            ViewDimension::D2 => Self::D2,
            ViewDimension::D3 => Self::D3,
            ViewDimension::Cube => Self::Cube,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    R8Unorm,
    R8G8B8A8Unorm,
    R16G16B16A16Float,
    BC1Unorm,
    BC2Unorm,
    BC3Unorm,
    BC4Unorm,
    BC5Unorm,
    BC7Unorm,
    B8G8R8A8Unorm,
}

impl From<xc3_model::ImageFormat> for ImageFormat {
    fn from(value: xc3_model::ImageFormat) -> Self {
        match value {
            xc3_model::ImageFormat::R8Unorm => Self::R8Unorm,
            xc3_model::ImageFormat::R8G8B8A8Unorm => Self::R8G8B8A8Unorm,
            xc3_model::ImageFormat::R16G16B16A16Float => Self::R16G16B16A16Float,
            xc3_model::ImageFormat::BC1Unorm => Self::BC1Unorm,
            xc3_model::ImageFormat::BC2Unorm => Self::BC2Unorm,
            xc3_model::ImageFormat::BC3Unorm => Self::BC3Unorm,
            xc3_model::ImageFormat::BC4Unorm => Self::BC4Unorm,
            xc3_model::ImageFormat::BC5Unorm => Self::BC5Unorm,
            xc3_model::ImageFormat::BC7Unorm => Self::BC7Unorm,
            xc3_model::ImageFormat::B8G8R8A8Unorm => Self::B8G8R8A8Unorm,
        }
    }
}

impl From<ImageFormat> for xc3_model::ImageFormat {
    fn from(value: ImageFormat) -> Self {
        match value {
            ImageFormat::R8Unorm => Self::R8Unorm,
            ImageFormat::R8G8B8A8Unorm => Self::R8G8B8A8Unorm,
            ImageFormat::R16G16B16A16Float => Self::R16G16B16A16Float,
            ImageFormat::BC1Unorm => Self::BC1Unorm,
            ImageFormat::BC2Unorm => Self::BC2Unorm,
            ImageFormat::BC3Unorm => Self::BC3Unorm,
            ImageFormat::BC4Unorm => Self::BC4Unorm,
            ImageFormat::BC5Unorm => Self::BC5Unorm,
            ImageFormat::BC7Unorm => Self::BC7Unorm,
            ImageFormat::B8G8R8A8Unorm => Self::B8G8R8A8Unorm,
        }
    }
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Sampler {
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mipmaps: bool,
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

impl From<xc3_model::AddressMode> for AddressMode {
    fn from(value: xc3_model::AddressMode) -> Self {
        match value {
            xc3_model::AddressMode::ClampToEdge => Self::ClampToEdge,
            xc3_model::AddressMode::Repeat => Self::Repeat,
            xc3_model::AddressMode::MirrorRepeat => Self::MirrorRepeat,
        }
    }
}

impl From<AddressMode> for xc3_model::AddressMode {
    fn from(value: AddressMode) -> Self {
        match value {
            AddressMode::ClampToEdge => Self::ClampToEdge,
            AddressMode::Repeat => Self::Repeat,
            AddressMode::MirrorRepeat => Self::MirrorRepeat,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub enum FilterMode {
    Nearest,
    Linear,
}

impl From<xc3_model::FilterMode> for FilterMode {
    fn from(value: xc3_model::FilterMode) -> Self {
        match value {
            xc3_model::FilterMode::Nearest => Self::Nearest,
            xc3_model::FilterMode::Linear => Self::Linear,
        }
    }
}

impl From<FilterMode> for xc3_model::FilterMode {
    fn from(value: FilterMode) -> Self {
        match value {
            FilterMode::Nearest => Self::Nearest,
            FilterMode::Linear => Self::Linear,
        }
    }
}

#[pymethods]
impl ModelRoot {
    pub fn decode_images_rgbaf32(&self, py: Python) -> Vec<PyObject> {
        // TODO: make decoding optional?
        // TODO: Expose xc3_model types from root?
        let buffers: Vec<_> = self
            .image_textures
            .par_iter()
            .map(|image| {
                // TODO: Use image_dds directly to avoid cloning?
                let bytes = xc3_model::ImageTexture {
                    name: image.name.clone(),
                    width: image.width,
                    height: image.height,
                    depth: image.depth,
                    view_dimension: image.view_dimension.into(),
                    image_format: image.image_format.into(),
                    mipmap_count: image.mipmap_count,
                    image_data: image.image_data.clone(),
                }
                .to_image()
                .unwrap()
                .into_raw();

                bytes
                    .into_iter()
                    .map(|u| u as f32 / 255.0)
                    .collect::<Vec<_>>()
            })
            .collect();

        buffers
            .into_iter()
            .map(|buffer| buffer.into_pyarray(py).into())
            .collect()
    }
}

#[pyfunction]
fn load_model(py: Python, wimdo_path: &str, database_path: Option<&str>) -> PyResult<ModelRoot> {
    let database = database_path.map(xc3_model::GBufferDatabase::from_file);
    let root = xc3_model::load_model(wimdo_path, database.as_ref());
    Ok(model_root(py, root))
}

#[pyfunction]
fn load_map(
    py: Python,
    wismhd_path: &str,
    database_path: Option<&str>,
) -> PyResult<Vec<ModelRoot>> {
    let database = database_path.map(xc3_model::GBufferDatabase::from_file);
    let roots = xc3_model::load_map(wismhd_path, database.as_ref());
    Ok(roots.into_iter().map(|root| model_root(py, root)).collect())
}

fn model_root(py: Python, root: xc3_model::ModelRoot) -> ModelRoot {
    ModelRoot {
        groups: root
            .groups
            .into_iter()
            .map(|group| ModelGroup {
                models: group
                    .models
                    .into_iter()
                    .map(|models| Models {
                        models: models.models.into_iter().map(|m| model(py, m)).collect(),
                        materials: materials(models.materials),
                        samplers: samplers(models.samplers),
                        skeleton: models.skeleton.map(|skeleton| Skeleton {
                            bones: skeleton
                                .bones
                                .into_iter()
                                .map(|bone| Bone {
                                    name: bone.name,
                                    transform: mat4_pyarray(py, bone.transform),
                                    parent_index: bone.parent_index,
                                })
                                .collect(),
                        }),
                        base_lod_indices: models.base_lod_indices,
                    })
                    .collect(),
                buffers: group
                    .buffers
                    .into_iter()
                    .map(|buffer| ModelBuffers {
                        vertex_buffers: vertex_buffers(py, buffer.vertex_buffers),
                        index_buffers: index_buffers(py, buffer.index_buffers),
                    })
                    .collect(),
            })
            .collect(),
        image_textures: root
            .image_textures
            .into_iter()
            .map(|image| ImageTexture {
                name: image.name,
                width: image.width,
                height: image.height,
                depth: image.depth,
                view_dimension: image.view_dimension.into(),
                image_format: image.image_format.into(),
                mipmap_count: image.mipmap_count,
                image_data: image.image_data,
            })
            .collect(),
    }
}

fn model(py: Python, model: xc3_model::Model) -> Model {
    Model {
        meshes: model
            .meshes
            .into_iter()
            .map(|mesh| Mesh {
                vertex_buffer_index: mesh.vertex_buffer_index,
                index_buffer_index: mesh.index_buffer_index,
                material_index: mesh.material_index,
                lod: mesh.lod,
            })
            .collect(),
        instances: transforms_pyarray(py, &model.instances),
        model_buffers_index: model.model_buffers_index,
    }
}

fn materials(materials: Vec<xc3_model::Material>) -> Vec<Material> {
    materials
        .into_iter()
        .map(|material| Material {
            name: material.name,
            textures: material
                .textures
                .into_iter()
                .map(|texture| Texture {
                    image_texture_index: texture.image_texture_index,
                })
                .collect(),
            alpha_test: material.alpha_test.map(|a| TextureAlphaTest {
                texture_index: a.texture_index,
                channel_index: a.channel_index,
                ref_value: a.ref_value,
            }),
            shader: material.shader.map(Shader),
            parameters: MaterialParameters {
                mat_color: material.parameters.mat_color.into(),
                alpha_test_ref: material.parameters.alpha_test_ref,
                tex_matrix: material.parameters.tex_matrix,
                work_float4: material
                    .parameters
                    .work_float4
                    .map(|v| v.into_iter().map(|v| v.into()).collect()),
                work_color: material
                    .parameters
                    .work_color
                    .map(|v| v.into_iter().map(|v| v.into()).collect()),
            },
        })
        .collect()
}

fn samplers(samplers: Vec<xc3_model::Sampler>) -> Vec<Sampler> {
    samplers
        .into_iter()
        .map(|sampler| Sampler {
            address_mode_u: sampler.address_mode_u.into(),
            address_mode_v: sampler.address_mode_v.into(),
            address_mode_w: sampler.address_mode_w.into(),
            min_filter: sampler.min_filter.into(),
            mag_filter: sampler.mag_filter.into(),
            mipmaps: sampler.mipmaps,
        })
        .collect()
}

fn vertex_buffers(py: Python, vertex_buffers: Vec<xc3_model::VertexBuffer>) -> Vec<VertexBuffer> {
    vertex_buffers
        .into_iter()
        .map(|buffer| VertexBuffer {
            attributes: vertex_attributes(py, buffer.attributes),
            influences: buffer
                .influences
                .into_iter()
                .map(|influence| Influence {
                    bone_name: influence.bone_name,
                    weights: influence
                        .weights
                        .into_iter()
                        .map(|weight| SkinWeight {
                            vertex_index: weight.vertex_index,
                            weight: weight.weight,
                        })
                        .collect(),
                })
                .collect(),
        })
        .collect()
}

fn vertex_attributes(
    py: Python,
    attributes: Vec<xc3_model::vertex::AttributeData>,
) -> Vec<AttributeData> {
    attributes
        .into_iter()
        .map(|attribute| match attribute {
            xc3_model::vertex::AttributeData::Position(values) => AttributeData {
                attribute_type: AttributeType::Position,
                data: vec3_pyarray(py, &values),
            },
            xc3_model::vertex::AttributeData::Normal(values) => AttributeData {
                attribute_type: AttributeType::Normal,
                data: vec4_pyarray(py, &values),
            },
            xc3_model::vertex::AttributeData::Tangent(values) => AttributeData {
                attribute_type: AttributeType::Tangent,
                data: vec4_pyarray(py, &values),
            },
            xc3_model::vertex::AttributeData::Uv1(values) => AttributeData {
                attribute_type: AttributeType::Uv1,
                data: vec2_pyarray(py, &values),
            },
            xc3_model::vertex::AttributeData::Uv2(values) => AttributeData {
                attribute_type: AttributeType::Uv2,
                data: vec2_pyarray(py, &values),
            },
            xc3_model::vertex::AttributeData::VertexColor(values) => AttributeData {
                attribute_type: AttributeType::VertexColor,
                data: vec4_pyarray(py, &values),
            },
            xc3_model::vertex::AttributeData::WeightIndex(values) => AttributeData {
                attribute_type: AttributeType::WeightIndex,
                data: values.into_pyarray(py).into(),
            },
            xc3_model::vertex::AttributeData::SkinWeights(values) => AttributeData {
                attribute_type: AttributeType::SkinWeights,
                data: vec4_pyarray(py, &values),
            },
            xc3_model::vertex::AttributeData::BoneIndices(values) => AttributeData {
                attribute_type: AttributeType::BoneIndices,
                data: uvec4_pyarray(py, &values),
            },
        })
        .collect()
}

fn index_buffers(py: Python, index_buffers: Vec<xc3_model::IndexBuffer>) -> Vec<IndexBuffer> {
    index_buffers
        .into_iter()
        .map(|buffer| IndexBuffer {
            indices: buffer.indices.into_pyarray(py).into(),
        })
        .collect()
}

fn vec2_pyarray(py: Python, values: &[Vec2]) -> PyObject {
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

fn vec3_pyarray(py: Python, values: &[Vec3]) -> PyObject {
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

fn vec4_pyarray(py: Python, values: &[Vec4]) -> PyObject {
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

// TODO: test cases for this.
fn mat4_pyarray(py: Python, transform: Mat4) -> PyObject {
    PyArray::from_slice(py, &transform.to_cols_array())
        .reshape((4, 4))
        .unwrap()
        .into()
}

#[pymodule]
fn xc3_model_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // TODO: automate registering every type?
    m.add_class::<ModelRoot>()?;
    m.add_class::<ModelGroup>()?;
    m.add_class::<ModelBuffers>()?;
    m.add_class::<Models>()?;
    m.add_class::<Model>()?;
    m.add_class::<Mesh>()?;
    m.add_class::<Skeleton>()?;
    m.add_class::<Bone>()?;
    m.add_class::<Material>()?;
    m.add_class::<TextureAlphaTest>()?;
    m.add_class::<MaterialParameters>()?;
    m.add_class::<Shader>()?;
    m.add_class::<BufferParameter>()?;
    m.add_class::<Texture>()?;
    m.add_class::<VertexBuffer>()?;
    m.add_class::<AttributeData>()?;
    m.add_class::<AttributeType>()?;
    m.add_class::<Influence>()?;
    m.add_class::<SkinWeight>()?;
    m.add_class::<IndexBuffer>()?;
    m.add_class::<ImageTexture>()?;
    m.add_class::<ViewDimension>()?;
    m.add_class::<ImageFormat>()?;
    m.add_class::<Sampler>()?;
    m.add_class::<AddressMode>()?;
    m.add_class::<FilterMode>()?;

    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(load_map, m)?)?;

    Ok(())
}

// TODO: Test cases for certain conversions?
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {}
}
