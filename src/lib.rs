use glam::{Mat4, Vec2, Vec3, Vec4};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use rayon::prelude::*;
use xc3_model::animation::BoneIndex;

// TODO: Match the module structure of xc3_model as closely as possible.
// Create a Python class for every public xc3_model type.
// We don't define these conversions on the xc3_model types themselves.
// This flexibility allows more efficient and idiomatic bindings.

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
    pub weights: Option<Weights>,
}

// TODO: Add methods to convert to influences.

#[pyclass]
#[derive(Debug, Clone)]
pub struct Weights {
    #[pyo3(get)]
    pub skin_weights: SkinWeights,
    // TODO: how to handle this?
    weight_groups: Vec<xc3_lib::vertex::WeightGroup>,
    weight_lods: Vec<xc3_lib::vertex::WeightLod>,
}

#[pymethods]
impl Weights {
    // TODO: Is it worth including the weight_group method?
    pub fn weights_start_index(
        &self,
        skin_flags: u32,
        lod: u16,
        unk_type: RenderPassType,
    ) -> usize {
        // TODO: find a cleaner way of doing this.
        xc3_model::Weights {
            skin_weights: xc3_model::skinning::SkinWeights {
                bone_indices: Vec::new(),
                weights: Vec::new(),
                bone_names: Vec::new(),
            },
            weight_groups: self.weight_groups.clone(),
            weight_lods: self.weight_lods.clone(),
        }
        .weights_start_index(skin_flags, lod, unk_type.into())
    }
}

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct SkinWeights {
    // N x 4 numpy.ndarray
    pub bone_indices: PyObject,
    // N x 4 numpy.ndarray
    pub weights: PyObject,
    /// The name list for the indices in [bone_indices](#structfield.bone_indices).
    pub bone_names: Vec<String>,
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
    pub skin_flags: u32,
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
    pub unk_type: RenderPassType,
    pub parameters: MaterialParameters,
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
    pub tex_matrix: Option<Vec<[f32; 16]>>,
    pub work_float4: Option<Vec<(f32, f32, f32, f32)>>,
    pub work_color: Option<Vec<(f32, f32, f32, f32)>>,
}

// TODO: Expose implementation details?
#[pyclass]
#[derive(Debug, Clone)]
pub struct Shader(xc3_model::shader_database::Shader);

#[pymethods]
impl Shader {
    pub fn sampler_channel_index(
        &self,
        output_index: usize,
        channel: char,
    ) -> Option<(usize, usize)> {
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
    pub attributes: Vec<AttributeData>,
    pub morph_targets: Vec<MorphTarget>,
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

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct MorphTarget {
    // N x 3 numpy.ndarray
    pub position_deltas: PyObject,
    // N x 4 numpy.ndarray
    pub normal_deltas: PyObject,
    // N x 4 numpy.ndarray
    pub tangent_deltas: PyObject,
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

// TODO: export texture usage from xc3_model?
#[pyclass(get_all)]
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

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct TextureUsage(pub u32);

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

python_enum!(
    AddressMode,
    xc3_model::AddressMode,
    ClampToEdge,
    Repeat,
    MirrorRepeat
);

python_enum!(FilterMode, xc3_model::FilterMode, Nearest, Linear);

#[pyclass(get_all)]
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

// TODO: Expose implementation details?
#[pyclass]
#[derive(Debug, Clone)]
pub struct Track(xc3_model::animation::Track);

#[pyclass(get_all)]
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

#[pymethods]
impl Animation {
    pub fn current_frame(&self, current_time_seconds: f32) -> f32 {
        // TODO: looping?
        current_time_seconds * self.frames_per_second
    }
}

#[pymethods]
impl Track {
    pub fn sample_translation(&self, frame: f32) -> (f32, f32, f32) {
        self.0.sample_translation(frame).into()
    }

    pub fn sample_rotation(&self, frame: f32) -> (f32, f32, f32, f32) {
        self.0.sample_rotation(frame).into()
    }

    pub fn sample_scale(&self, frame: f32) -> (f32, f32, f32) {
        self.0.sample_scale(frame).into()
    }

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
                    usage: None, // TODO: add to Python bindings
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

// TODO: avoid unwrap?
#[pyfunction]
fn load_model(py: Python, wimdo_path: &str, database_path: Option<&str>) -> PyResult<ModelRoot> {
    let database =
        database_path.map(|p| xc3_model::shader_database::ShaderDatabase::from_file(p).unwrap());
    let root = xc3_model::load_model(wimdo_path, database.as_ref()).unwrap();
    Ok(model_root(py, root))
}

#[pyfunction]
fn load_map(
    py: Python,
    wismhd_path: &str,
    database_path: Option<&str>,
) -> PyResult<Vec<ModelRoot>> {
    let database =
        database_path.map(|p| xc3_model::shader_database::ShaderDatabase::from_file(p).unwrap());
    let roots = xc3_model::load_map(wismhd_path, database.as_ref()).unwrap();
    Ok(roots.into_iter().map(|root| model_root(py, root)).collect())
}

#[pyfunction]
fn load_animations(_py: Python, anim_path: &str) -> PyResult<Vec<Animation>> {
    let animations = xc3_model::load_animations(anim_path).unwrap();
    Ok(animations.into_iter().map(animation).collect())
}

#[pyfunction]
fn murmur3(name: &str) -> u32 {
    xc3_model::animation::murmur3(name.as_bytes())
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
                        weights: buffer.weights.map(|weights| Weights {
                            skin_weights: SkinWeights {
                                bone_indices: uvec4_pyarray(py, &weights.skin_weights.bone_indices),
                                weights: vec4_pyarray(py, &weights.skin_weights.weights),
                                bone_names: weights.skin_weights.bone_names,
                            },
                            weight_groups: weights.weight_groups,
                            weight_lods: weights.weight_lods,
                        }),
                    })
                    .collect(),
            })
            .collect(),
        image_textures: root
            .image_textures
            .into_iter()
            .map(|image| ImageTexture {
                name: image.name,
                usage: image.usage.map(|u| TextureUsage(u as u32)),
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
                skin_flags: mesh.skin_flags,
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
            unk_type: material.unk_type.into(),
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
            morph_targets: morph_targets(py, buffer.morph_targets),
        })
        .collect()
}

fn vertex_attributes(
    py: Python,
    attributes: Vec<xc3_model::vertex::AttributeData>,
) -> Vec<AttributeData> {
    attributes
        .into_iter()
        .map(|attribute| attribute_data(py, attribute))
        .collect()
}

fn attribute_data(py: Python, attribute: xc3_model::vertex::AttributeData) -> AttributeData {
    match attribute {
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
        xc3_model::vertex::AttributeData::TexCoord0(values) => AttributeData {
            attribute_type: AttributeType::TexCoord0,
            data: vec2_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord1(values) => AttributeData {
            attribute_type: AttributeType::TexCoord1,
            data: vec2_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord2(values) => AttributeData {
            attribute_type: AttributeType::TexCoord2,
            data: vec2_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord3(values) => AttributeData {
            attribute_type: AttributeType::TexCoord3,
            data: vec2_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord4(values) => AttributeData {
            attribute_type: AttributeType::TexCoord4,
            data: vec2_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord5(values) => AttributeData {
            attribute_type: AttributeType::TexCoord5,
            data: vec2_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord6(values) => AttributeData {
            attribute_type: AttributeType::TexCoord6,
            data: vec2_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord7(values) => AttributeData {
            attribute_type: AttributeType::TexCoord7,
            data: vec2_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::TexCoord8(values) => AttributeData {
            attribute_type: AttributeType::TexCoord8,
            data: vec2_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::VertexColor(values) => AttributeData {
            attribute_type: AttributeType::VertexColor,
            data: vec4_pyarray(py, &values),
        },
        xc3_model::vertex::AttributeData::Blend(values) => AttributeData {
            attribute_type: AttributeType::Blend,
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
    }
}

fn morph_targets(py: Python, targets: Vec<xc3_model::MorphTarget>) -> Vec<MorphTarget> {
    targets
        .into_iter()
        .map(|target| MorphTarget {
            position_deltas: vec3_pyarray(py, &target.position_deltas),
            normal_deltas: vec4_pyarray(py, &target.normal_deltas),
            tangent_deltas: vec4_pyarray(py, &target.tangent_deltas),
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

fn animation(animation: xc3_model::animation::Animation) -> Animation {
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
    // TODO: split into submodules?
    m.add_class::<ModelRoot>()?;
    m.add_class::<ModelGroup>()?;
    m.add_class::<ModelBuffers>()?;
    m.add_class::<Weights>()?;
    m.add_class::<SkinWeights>()?;
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
    m.add_class::<MorphTarget>()?;
    m.add_class::<Influence>()?;
    m.add_class::<SkinWeight>()?;
    m.add_class::<IndexBuffer>()?;

    m.add_class::<ImageTexture>()?;
    m.add_class::<TextureUsage>()?;
    m.add_class::<ViewDimension>()?;
    m.add_class::<ImageFormat>()?;

    m.add_class::<Sampler>()?;
    m.add_class::<AddressMode>()?;
    m.add_class::<FilterMode>()?;

    m.add_class::<Animation>()?;
    m.add_class::<Track>()?;
    m.add_class::<Keyframe>()?;
    m.add_class::<SpaceMode>()?;
    m.add_class::<PlayMode>()?;
    m.add_class::<BlendMode>()?;

    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(load_map, m)?)?;
    m.add_function(wrap_pyfunction!(load_animations, m)?)?;
    m.add_function(wrap_pyfunction!(murmur3, m)?)?;

    Ok(())
}

// TODO: Test cases for certain conversions?
#[cfg(test)]
mod tests {}
