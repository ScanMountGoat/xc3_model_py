use pyo3::prelude::*;

#[pymodule]
pub mod monolib {
    use pyo3::prelude::*;

    use crate::{map_py::MapPy, map_py_wrapper_impl, xc3_model_py::ImageTexture};

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct ShaderTextures(pub xc3_model::monolib::ShaderTextures);

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
}
