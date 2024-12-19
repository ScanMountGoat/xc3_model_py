use pyo3::prelude::*;

#[pymodule]
pub mod collision {
    use pyo3::{prelude::*, types::PyList};

    use crate::map_py::MapPy;

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::collision::CollisionMeshes)]
    pub struct CollisionMeshes {
        pub vertices: PyObject,
        pub meshes: Py<PyList>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::collision::CollisionMesh)]
    pub struct CollisionMesh {
        pub name: String,
        pub instances: PyObject,
        pub indices: PyObject,
    }
}
