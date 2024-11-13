use crate::python_enum;
use pyo3::prelude::*;

python_enum!(
    LayerBlendMode,
    xc3_model::shader_database::LayerBlendMode,
    Mix,
    MixRatio,
    Add,
    AddNormal,
    Overlay
);

#[pymodule]
pub mod shader_database {

    use indexmap::IndexMap;
    use pyo3::{
        prelude::*,
        types::{PyDict, PyList},
    };
    use smol_str::SmolStr;

    use crate::{map_py::MapPy, map_py_wrapper_impl, py_exception};

    #[pymodule_export]
    use super::LayerBlendMode;

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct ShaderDatabase(pub xc3_model::shader_database::ShaderDatabase);

    #[pymethods]
    impl ShaderDatabase {
        #[staticmethod]
        pub fn from_file(path: &str) -> PyResult<Self> {
            Ok(Self(
                xc3_model::shader_database::ShaderDatabase::from_file(path)
                    .map_err(py_exception)?,
            ))
        }

        pub fn model(&self, py: Python, name: &str) -> PyResult<Option<ModelPrograms>> {
            self.0.model(name).map(|m| m.map_py(py)).transpose()
        }

        pub fn map(&self, py: Python, name: &str) -> PyResult<Option<MapPrograms>> {
            self.0.map(name).map(|m| m.map_py(py)).transpose()
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::ModelPrograms)]
    pub struct ModelPrograms {
        pub programs: Py<PyList>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::MapPrograms)]
    pub struct MapPrograms {
        pub map_models: Py<PyList>,
        pub prop_models: Py<PyList>,
        pub env_models: Py<PyList>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::ShaderProgram)]
    pub struct ShaderProgram {
        pub output_dependencies: Py<PyDict>,
        pub outline_width: Option<Dependency>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::OutputDependencies)]
    pub struct OutputDependencies {
        pub dependencies: Py<PyList>,
        pub layers: Py<PyList>,
    }

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct Dependency(xc3_model::shader_database::Dependency);

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::BufferDependency)]
    pub struct BufferDependency {
        pub name: String,
        pub field: String,
        pub index: Option<usize>,
        pub channel: Option<char>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::TextureDependency)]
    pub struct TextureDependency {
        pub name: String,
        pub channel: Option<char>,
        pub texcoords: Py<PyList>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::TexCoord)]
    pub struct TexCoord {
        pub name: String,
        pub channel: Option<char>,
        pub params: Option<Py<TexCoordParams>>,
    }

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct TexCoordParams(xc3_model::shader_database::TexCoordParams);

    // Workaround for representing Rust enums in Python.
    #[pymethods]
    impl TexCoordParams {
        pub fn scale(&self, py: Python) -> PyResult<Option<BufferDependency>> {
            match &self.0 {
                xc3_model::shader_database::TexCoordParams::Scale(b) => b.map_py(py).map(Some),
                _ => Ok(None),
            }
        }

        pub fn matrix(&self, py: Python) -> PyResult<Option<[BufferDependency; 4]>> {
            match &self.0 {
                xc3_model::shader_database::TexCoordParams::Matrix(m) => m.map_py(py).map(Some),
                _ => Ok(None),
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::AttributeDependency)]
    pub struct AttributeDependency {
        pub name: String,
        pub channel: Option<char>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::TextureLayer)]
    pub struct TextureLayer {
        pub value: Dependency,
        pub ratio: Option<Py<Dependency>>,
        pub blend_mode: LayerBlendMode,
        pub is_fresnel: bool,
    }

    // Workaround for representing Rust enums in Python.
    #[pymethods]
    impl Dependency {
        pub fn constant(&self) -> Option<f32> {
            match &self.0 {
                xc3_model::shader_database::Dependency::Constant(c) => Some(c.0),
                _ => None,
            }
        }

        pub fn buffer(&self, py: Python) -> PyResult<Option<BufferDependency>> {
            match &self.0 {
                xc3_model::shader_database::Dependency::Buffer(b) => b.map_py(py).map(Some),
                _ => Ok(None),
            }
        }

        pub fn texture(&self, py: Python) -> PyResult<Option<TextureDependency>> {
            match &self.0 {
                xc3_model::shader_database::Dependency::Texture(t) => t.map_py(py).map(Some),
                _ => Ok(None),
            }
        }

        pub fn attribute(&self, py: Python) -> PyResult<Option<AttributeDependency>> {
            match &self.0 {
                xc3_model::shader_database::Dependency::Attribute(a) => a.map_py(py).map(Some),
                _ => Ok(None),
            }
        }
    }

    impl MapPy<Py<PyDict>> for IndexMap<SmolStr, xc3_model::shader_database::OutputDependencies> {
        fn map_py(&self, py: Python) -> PyResult<Py<PyDict>> {
            let dict = PyDict::new_bound(py);
            for (k, v) in self.iter() {
                let v: OutputDependencies = v.map_py(py)?;
                dict.set_item(k.to_string(), v.into_py(py))?;
            }
            Ok(dict.into())
        }
    }

    impl MapPy<IndexMap<SmolStr, xc3_model::shader_database::OutputDependencies>> for Py<PyDict> {
        fn map_py(
            &self,
            py: Python,
        ) -> PyResult<IndexMap<SmolStr, xc3_model::shader_database::OutputDependencies>> {
            self.extract::<IndexMap<String, OutputDependencies>>(py)?
                .into_iter()
                .map(|(k, v)| Ok((k.into(), v.map_py(py)?)))
                .collect()
        }
    }

    map_py_wrapper_impl!(xc3_model::shader_database::TexCoordParams, TexCoordParams);
    map_py_wrapper_impl!(xc3_model::shader_database::Dependency, Dependency);
}
