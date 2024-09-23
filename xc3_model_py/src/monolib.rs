use pyo3::prelude::*;

use crate::{map_py::MapPy, map_py_wrapper_impl, ImageTexture};

#[pyclass]
#[derive(Debug, Clone)]
pub struct ShaderTextures(xc3_model::monolib::ShaderTextures);

#[pymethods]
impl ShaderTextures {
    #[staticmethod]
    fn from_folder(path: &str) -> Self {
        Self(xc3_model::monolib::ShaderTextures::from_folder(path))
    }

    fn global_texture(&self, py: Python, sampler_name: &str) -> PyResult<Option<ImageTexture>> {
        self.0
            .global_texture(sampler_name)
            .map(|t| t.map_py(py))
            .transpose()
    }
}

map_py_wrapper_impl!(xc3_model::monolib::ShaderTextures, ShaderTextures);

pub fn monolib(py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "monolib")?;

    m.add_class::<ShaderTextures>()?;

    module.add_submodule(&m)?;
    Ok(())
}
