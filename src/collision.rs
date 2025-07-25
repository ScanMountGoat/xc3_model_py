use pyo3::prelude::*;

#[pymodule]
pub mod collision {
    use numpy::{PyArray1, PyArray2, PyArray3};
    use pyo3::prelude::*;

    use map_py::{MapPy, TypedList};

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::collision::CollisionMeshes)]
    pub struct CollisionMeshes {
        pub vertices: Py<PyArray2<f32>>,
        pub meshes: TypedList<CollisionMesh>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::collision::CollisionMesh)]
    pub struct CollisionMesh {
        pub name: String,
        pub instances: Py<PyArray3<f32>>,
        pub indices: Py<PyArray1<u32>>,
    }
}
