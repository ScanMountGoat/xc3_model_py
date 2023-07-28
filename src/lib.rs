use std::collections::HashMap;

use glam::{Mat4, Vec2, Vec3, Vec4};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;

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
    pub models: Vec<Model>,
    pub materials: Vec<Material>,
    pub skeleton: Option<Skeleton>,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub vertex_buffers: Vec<VertexBuffer>,
    pub index_buffers: Vec<IndexBuffer>,
    // N x 4 x 4 numpy.ndarray
    pub instances: PyObject,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertex_buffer_index: usize,
    pub index_buffer_index: usize,
    pub material_index: usize,
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
    // pub flags: MaterialFlags,
    pub textures: Vec<Texture>,
    pub shader: Option<Shader>,
    // pub unk_type: ShaderUnkType,
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct Shader {
    // TODO: Why can't we use indexmap here?
    pub output_dependencies: HashMap<String, Vec<String>>,
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

#[pyclass]
#[derive(Debug, Clone)]
pub enum ViewDimension {
    D2 = 1,
    D3 = 2,
    Cube = 8,
}

impl From<xc3_model::texture::ViewDimension> for ViewDimension {
    fn from(value: xc3_model::texture::ViewDimension) -> Self {
        match value {
            xc3_model::texture::ViewDimension::D2 => Self::D2,
            xc3_model::texture::ViewDimension::D3 => Self::D3,
            xc3_model::texture::ViewDimension::Cube => Self::Cube,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum ImageFormat {
    R8Unorm = 1,
    R8G8B8A8Unorm = 37,
    R16G16B16A16Float = 41,
    BC1Unorm = 66,
    BC2Unorm = 67,
    BC3Unorm = 68,
    BC4Unorm = 73,
    BC5Unorm = 75,
    BC7Unorm = 77,
    B8G8R8A8Unorm = 109,
}

impl From<xc3_model::texture::ImageFormat> for ImageFormat {
    fn from(value: xc3_model::texture::ImageFormat) -> Self {
        match value {
            xc3_model::texture::ImageFormat::R8Unorm => Self::R8Unorm,
            xc3_model::texture::ImageFormat::R8G8B8A8Unorm => Self::R8G8B8A8Unorm,
            xc3_model::texture::ImageFormat::R16G16B16A16Float => Self::R16G16B16A16Float,
            xc3_model::texture::ImageFormat::BC1Unorm => Self::BC1Unorm,
            xc3_model::texture::ImageFormat::BC2Unorm => Self::BC2Unorm,
            xc3_model::texture::ImageFormat::BC3Unorm => Self::BC3Unorm,
            xc3_model::texture::ImageFormat::BC4Unorm => Self::BC4Unorm,
            xc3_model::texture::ImageFormat::BC5Unorm => Self::BC5Unorm,
            xc3_model::texture::ImageFormat::BC7Unorm => Self::BC7Unorm,
            xc3_model::texture::ImageFormat::B8G8R8A8Unorm => Self::B8G8R8A8Unorm,
        }
    }
}

#[pyfunction]
fn load_model(
    py: Python,
    wimdo_path: &str,
    // TODO: database?
) -> PyResult<ModelRoot> {
    let root = xc3_model::load_model(wimdo_path, None);
    Ok(model_root(py, root))
}

#[pyfunction]
fn load_map(
    py: Python,
    wismhd_path: &str,
    // TODO: database?
) -> PyResult<Vec<ModelRoot>> {
    let roots = xc3_model::load_map(wismhd_path, None);
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
                    .map(|model| Model {
                        meshes: model
                            .meshes
                            .into_iter()
                            .map(|mesh| Mesh {
                                vertex_buffer_index: mesh.vertex_buffer_index,
                                index_buffer_index: mesh.index_buffer_index,
                                material_index: mesh.material_index,
                            })
                            .collect(),
                        vertex_buffers: vertex_buffers(py, model.vertex_buffers),
                        index_buffers: index_buffers(py, model.index_buffers),
                        instances: transforms_pyarray(py, &model.instances),
                    })
                    .collect(),
                materials: materials(group.materials),
                skeleton: group.skeleton.map(|skeleton| Skeleton {
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
            shader: material.shader.map(|shader| Shader {
                output_dependencies: shader.output_dependencies.into_iter().collect(),
            }),
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
        .into_iter()
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
        .into_iter()
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
        .into_iter()
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
        .into_iter()
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
        .into_iter()
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
    m.add_class::<Model>()?;
    m.add_class::<Mesh>()?;
    m.add_class::<Skeleton>()?;
    m.add_class::<Bone>()?;
    m.add_class::<Material>()?;
    m.add_class::<Shader>()?;
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
