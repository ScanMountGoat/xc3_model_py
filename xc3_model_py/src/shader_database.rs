use indexmap::IndexMap;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use smol_str::SmolStr;

use crate::{map_py::MapPy, py_exception};

#[pyclass]
#[derive(Debug, Clone)]
pub struct ShaderDatabase(pub xc3_model::shader_database::ShaderDatabase);

#[pymethods]
impl ShaderDatabase {
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        Ok(Self(
            xc3_model::shader_database::ShaderDatabase::from_file(path).map_err(py_exception)?,
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
    pub index: usize,
    pub channels: String,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::shader_database::TextureDependency)]
pub struct TextureDependency {
    pub name: String,
    pub channels: String,
    pub texcoords: Py<PyList>,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::shader_database::TexCoord)]
pub struct TexCoord {
    pub name: String,
    pub channels: String,
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
    pub channels: String,
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

impl MapPy<Py<PyDict>> for IndexMap<SmolStr, Vec<xc3_model::shader_database::Dependency>> {
    fn map_py(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        for (k, v) in self.iter() {
            let values = PyList::new_bound(py, v.iter().map(|v| Dependency(v.clone()).into_py(py)));
            dict.set_item(k.to_string(), values)?;
        }
        Ok(dict.into())
    }
}

impl MapPy<IndexMap<SmolStr, Vec<xc3_model::shader_database::Dependency>>> for Py<PyDict> {
    fn map_py(
        &self,
        py: Python,
    ) -> PyResult<IndexMap<SmolStr, Vec<xc3_model::shader_database::Dependency>>> {
        Ok(self
            .extract::<IndexMap<String, Vec<Dependency>>>(py)?
            .into_iter()
            .map(|(k, v)| (k.into(), v.into_iter().map(|v| v.0).collect()))
            .collect())
    }
}

// TODO: macro for this?
impl MapPy<xc3_model::shader_database::TexCoordParams> for TexCoordParams {
    fn map_py(&self, py: Python) -> PyResult<xc3_model::shader_database::TexCoordParams> {
        Ok(self.0.clone())
    }
}

impl MapPy<TexCoordParams> for xc3_model::shader_database::TexCoordParams {
    fn map_py(&self, py: Python) -> PyResult<TexCoordParams> {
        Ok(TexCoordParams(self.clone()))
    }
}

// Map to and from Py<T>
impl MapPy<Py<TexCoordParams>> for xc3_model::shader_database::TexCoordParams {
    fn map_py(&self, py: Python) -> PyResult<Py<TexCoordParams>> {
        let value: TexCoordParams = self.map_py(py)?;
        Py::new(py, value)
    }
}

impl MapPy<xc3_model::shader_database::TexCoordParams> for Py<TexCoordParams> {
    fn map_py(&self, py: Python) -> PyResult<xc3_model::shader_database::TexCoordParams> {
        self.extract::<TexCoordParams>(py)?.map_py(py)
    }
}

pub fn shader_database(py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "shader_database")?;

    m.add_class::<ShaderDatabase>()?;
    m.add_class::<ModelPrograms>()?;
    m.add_class::<MapPrograms>()?;
    m.add_class::<ShaderProgram>()?;
    m.add_class::<Dependency>()?;
    m.add_class::<TextureDependency>()?;
    m.add_class::<TexCoord>()?;
    m.add_class::<TexCoordParams>()?;
    m.add_class::<AttributeDependency>()?;

    module.add_submodule(&m)?;
    Ok(())
}
