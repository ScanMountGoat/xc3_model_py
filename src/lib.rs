use numpy::PyArray3;
use pyo3::{create_exception, exceptions::PyException, prelude::*};
use rayon::prelude::*;

mod animation;
mod collision;
mod material;
mod monolib;
mod shader_database;
mod skinning;
mod vertex;

// Create a Python class for every public xc3_model type.
// We don't define these conversions on the xc3_model types themselves.
// This flexibility allows more efficient and idiomatic bindings.
// Py<PyList> creates a pure Python list for allowing mutating of elements.
// Py<PyArray> supports numpy.ndarray for performance.
// Vec<u8> uses Python's bytes object for performance.

create_exception!(xc3_model_py, Xc3ModelError, PyException);

#[macro_export]
macro_rules! python_enum {
    ($py_ty:ident, $rust_ty:ty, $( $i:ident ),+) => {
        #[pyclass(eq, eq_int)]
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

        impl ::map_py::MapPy<$rust_ty> for $py_ty {
            fn map_py(self, _py: Python) -> PyResult<$rust_ty> {
                Ok(self.into())
            }
        }

        impl ::map_py::MapPy<$py_ty> for $rust_ty {
            fn map_py(self, _py: Python) -> PyResult<$py_ty> {
                Ok(self.into())
            }
        }
    };
}

// TODO: traits and functions to simplify list conversions?

fn py_exception<E: Into<anyhow::Error>>(e: E) -> PyErr {
    // anyhow provides more detailed context for inner errors.
    PyErr::new::<Xc3ModelError, _>(format!("{:?}", anyhow::anyhow!(e)))
}

// TODO: test cases for conversions.

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

python_enum!(
    AddressMode,
    xc3_model::AddressMode,
    ClampToEdge,
    Repeat,
    MirrorRepeat
);

python_enum!(FilterMode, xc3_model::FilterMode, Nearest, Linear);

// Match the module hierarchy and types of xc3_model as closely as possible.
#[pymodule]
mod xc3_model_py {
    use super::*;

    use std::io::Cursor;
    use std::{ops::Deref, path::Path};

    use crate::material::TextureUsage;
    use crate::shader_database::shader_database::ShaderDatabase;
    use crate::skinning::skinning::Skinning;
    use crate::vertex::vertex::ModelBuffers;
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArrayDyn};
    use pyo3::types::PyBytes;
    use xc3_lib::dds::DdsExt;

    use map_py::{MapPy, TypedList};

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        Ok(())
    }

    #[pymodule_export]
    use animation::animation;

    #[pymodule_export]
    use collision::collision;

    #[pymodule_export]
    use material::material;

    #[pymodule_export]
    use monolib::monolib;

    #[pymodule_export]
    use shader_database::shader_database;

    #[pymodule_export]
    use skinning::skinning;

    #[pymodule_export]
    use vertex::vertex;

    #[pymodule_export]
    use super::ViewDimension;

    #[pymodule_export]
    use super::ImageFormat;

    #[pymodule_export]
    use super::AddressMode;

    #[pymodule_export]
    use super::FilterMode;

    #[pymodule_export]
    use super::Xc3ModelError;

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::ModelRoot)]
    pub struct ModelRoot {
        #[map(from(map_py::helpers::into_py), into(map_py::helpers::from_py))]
        pub models: Py<Models>,
        #[map(from(map_py::helpers::into_py), into(map_py::helpers::from_py))]
        pub buffers: Py<ModelBuffers>,
        pub image_textures: TypedList<ImageTexture>,
        #[map(
            from(map_py::helpers::into_option_py),
            into(map_py::helpers::from_option_py)
        )]
        pub skeleton: Option<Py<Skeleton>>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::MapRoot)]
    pub struct MapRoot {
        pub groups: TypedList<ModelGroup>,
        pub image_textures: TypedList<ImageTexture>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::ModelGroup)]
    pub struct ModelGroup {
        pub models: TypedList<Models>,
        pub buffers: TypedList<ModelBuffers>,
    }

    #[pymethods]
    impl ModelGroup {
        #[new]
        pub fn new(models: TypedList<Models>, buffers: TypedList<ModelBuffers>) -> Self {
            Self { models, buffers }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::Models)]
    pub struct Models {
        pub models: TypedList<Model>,
        pub materials: TypedList<crate::material::material::Material>,
        pub samplers: TypedList<Sampler>,
        #[map(
            from(map_py::helpers::into_option_py),
            into(map_py::helpers::from_option_py)
        )]
        pub skinning: Option<Py<Skinning>>,
        pub morph_controller_names: TypedList<String>,
        pub animation_morph_names: TypedList<String>,
        pub max_xyz: [f32; 3],
        pub min_xyz: [f32; 3],
        #[map(
            from(map_py::helpers::into_option_py),
            into(map_py::helpers::from_option_py)
        )]
        pub lod_data: Option<Py<LodData>>,
    }

    #[pymethods]
    impl Models {
        #[new]
        pub fn new(
            models: TypedList<Model>,
            materials: TypedList<crate::material::material::Material>,
            samplers: TypedList<Sampler>,
            max_xyz: [f32; 3],
            min_xyz: [f32; 3],
            morph_controller_names: TypedList<String>,
            animation_morph_names: TypedList<String>,
            skinning: Option<Py<Skinning>>,
            lod_data: Option<Py<LodData>>,
        ) -> Self {
            Self {
                models,
                materials,
                samplers,
                skinning,
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
        pub meshes: TypedList<Mesh>,
        // N x 4 x 4 numpy.ndarray
        pub instances: Py<PyArray3<f32>>,
        pub model_buffers_index: usize,
        pub max_xyz: [f32; 3],
        pub min_xyz: [f32; 3],
        pub bounding_radius: f32,
    }

    #[pymethods]
    impl Model {
        #[new]
        pub fn new(
            meshes: TypedList<Mesh>,
            instances: Py<PyArray3<f32>>,
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
        #[map(from(map_py::helpers::into), into(map_py::helpers::try_into))]
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
        pub items: TypedList<LodItem>,
        pub groups: TypedList<LodGroup>,
    }

    #[pymethods]
    impl LodData {
        #[new]
        pub fn new(items: TypedList<LodItem>, groups: TypedList<LodGroup>) -> Self {
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
        pub bones: TypedList<Bone>,
    }

    #[pymethods]
    impl Skeleton {
        // TODO: generate this with some sort of macro?
        #[new]
        fn new(bones: TypedList<Bone>) -> Self {
            Self { bones }
        }

        pub fn model_space_transforms(&self, py: Python) -> PyResult<Py<PyArray3<f32>>> {
            let transforms = self.clone().map_py(py)?.model_space_transforms();
            let matrices: Vec<_> = transforms.into_iter().map(|t| t.to_matrix()).collect();
            matrices.map_py(py)
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::Bone)]
    pub struct Bone {
        pub name: String,
        #[map(from(map_py::helpers::into_py), into(map_py::helpers::from_py))]
        pub transform: Py<Transform>,
        pub parent_index: Option<usize>,
    }

    #[pymethods]
    impl Bone {
        #[new]
        fn new(name: String, transform: Py<Transform>, parent_index: Option<usize>) -> Self {
            Self {
                name,
                transform,
                parent_index,
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

        #[staticmethod]
        fn from_dds(
            py: Python,
            dds: PyRef<Dds>,
            name: Option<String>,
            usage: Option<TextureUsage>,
        ) -> PyResult<Self> {
            xc3_model::ImageTexture::from_dds(&dds.0, name, usage.map(Into::into))
                .map_err(py_exception)?
                .map_py(py)
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
        pub data: Py<PyArray1<f32>>,
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
            data: Py<PyArray1<f32>>,
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

    #[pyclass]
    #[derive(Debug)]
    pub struct Dds(image_dds::ddsfile::Dds);

    #[pymethods]
    impl Dds {
        #[staticmethod]
        pub fn from_file(path: &str) -> PyResult<Self> {
            image_dds::ddsfile::Dds::from_file(path)
                .map(Dds)
                .map_err(py_exception)
        }

        pub fn save(&self, path: &str) -> PyResult<()> {
            self.0.save(path).map_err(py_exception)
        }
    }

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
            image_textures: TypedList<ImageTexture>,
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

        pub fn to_mxmd_model(
            &self,
            py: Python,
            mxmd: &Mxmd,
            msrd: &Msrd,
        ) -> PyResult<(Mxmd, Msrd)> {
            let (mxmd, msrd) = self
                .clone()
                .map_py(py)?
                .to_mxmd_model(&mxmd.0, &msrd.0)
                .map_err(py_exception)?;
            Ok((Mxmd(mxmd), Msrd(msrd)))
        }
        // TODO: support texture edits as well?
    }

    #[pymethods]
    impl MapRoot {
        #[new]
        pub fn new(groups: TypedList<ModelGroup>, image_textures: TypedList<ImageTexture>) -> Self {
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
        image_textures: &TypedList<ImageTexture>,
        folder: &str,
        prefix: &str,
        ext: &str,
        flip_vertical: bool,
    ) -> PyResult<Vec<String>> {
        let textures: Vec<xc3_model::ImageTexture> = image_textures.clone().map_py(py)?;
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

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::Transform)]
    pub struct Transform {
        pub translation: [f32; 3],
        pub rotation: [f32; 4],
        pub scale: [f32; 3],
    }

    #[pymethods]
    impl Transform {
        #[new]
        pub fn new(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
            Self {
                translation,
                rotation,
                scale,
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
    ) -> PyResult<TypedList<MapRoot>> {
        let database = shader_database.map(|database| &database.0);
        // Prevent Python from locking up while Rust processes map data in parallel.
        let roots = py.allow_threads(move || {
            xc3_model::load_map(wismhd_path, database).map_err(py_exception)
        })?;
        roots.map_py(py)
    }

    #[pyfunction]
    fn load_animations(py: Python, anim_path: &str) -> PyResult<TypedList<animation::Animation>> {
        let animations = xc3_model::load_animations(anim_path).map_err(py_exception)?;
        animations.map_py(py)
    }

    #[pyfunction]
    fn load_collisions(py: Python, idcm_path: &str) -> PyResult<collision::CollisionMeshes> {
        let collisions = xc3_model::load_collisions(idcm_path).map_err(py_exception)?;
        collisions.map_py(py)
    }

    #[pyfunction]
    fn decode_images_png(
        py: Python,
        image_textures: Vec<PyRef<ImageTexture>>,
        flip_vertical: bool,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let textures: Vec<&ImageTexture> = image_textures.iter().map(|i| i.deref()).collect();
        let buffers = textures
            .par_iter()
            .map(|image| {
                let format: xc3_model::ImageFormat = image.image_format.into();
                // TODO: expose to_surface in xc3_model.
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
                    .decode_layers_mipmaps_rgba8(0..surface.layers, 0..1)
                    .map_err(py_exception)?
                    .data)
            })
            .collect::<PyResult<Vec<_>>>()?;

        buffers
            .into_iter()
            .zip(textures)
            .map(|(buffer, texture)| {
                // TODO: avoid unwrap.
                let mut writer = Cursor::new(Vec::new());
                let mut image =
                    image_dds::image::RgbaImage::from_raw(texture.width, texture.height, buffer)
                        .unwrap();
                if flip_vertical {
                    image = image_dds::image::imageops::flip_vertical(&image);
                }
                image
                    .write_to(&mut writer, image_dds::image::ImageFormat::Png)
                    .unwrap();

                Ok(PyBytes::new(py, &writer.into_inner()).into())
            })
            .collect()
    }

    #[pyfunction]
    fn decode_images_rgbaf32(
        py: Python,
        image_textures: Vec<PyRef<ImageTexture>>,
    ) -> PyResult<Vec<Py<PyArray1<f32>>>> {
        let textures: Vec<&ImageTexture> = image_textures.iter().map(|i| i.deref()).collect();
        let buffers = textures
            .par_iter()
            .map(|image| {
                let format: xc3_model::ImageFormat = image.image_format.into();
                // TODO: expose to_surface in xc3_model.
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
            .map(|buffer| buffer.into_pyarray(py).into())
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
}
