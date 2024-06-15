use glam::{Vec2, Vec3, Vec4};
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
pub use xc3_model_py_derive::MapPy;

// Define a mapping between types.
// This allows for deriving the Python <-> Rust conversion.
// The derive macro is mainly to automate mapping field names.
pub trait MapPy<T> {
    fn map_py(&self, py: Python) -> PyResult<T>;
}

// Implement for primitive types.
macro_rules! map_py_impl {
    ($($t:ty),*) => {
        $(
            impl MapPy<$t> for $t {
                fn map_py(&self, _py: Python) -> PyResult<$t> {
                    Ok(self.clone())
                }
            }
        )*
    }
}

map_py_impl!(bool, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, f32, f64, String);

impl<T, U> MapPy<Option<U>> for Option<T>
where
    T: MapPy<U>,
{
    fn map_py(&self, py: Python) -> PyResult<Option<U>> {
        self.as_ref().map(|v| v.map_py(py)).transpose()
    }
}

// TODO: How to avoid overlapping definitions with numpy and Vec<T> for PyObject?
// TODO: how to implement for Py<T>?

impl MapPy<PyObject> for Vec<Vec2> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        vectors_pyarray(py, self)
    }
}

impl MapPy<Vec<Vec2>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<Vec2>> {
        pyarray_vectors(py, self)
    }
}

impl MapPy<PyObject> for Vec<Vec3> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        vectors_pyarray(py, self)
    }
}

impl MapPy<Vec<Vec3>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<Vec3>> {
        pyarray_vectors(py, self)
    }
}

impl MapPy<PyObject> for Vec<Vec4> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        vectors_pyarray(py, self)
    }
}

impl MapPy<Vec<Vec4>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<Vec4>> {
        pyarray_vectors(py, self)
    }
}

fn vectors_pyarray<const N: usize, T>(py: Python, values: &[T]) -> PyResult<PyObject>
where
    T: Into<[f32; N]> + Copy,
{
    // This flatten will be optimized in Release mode.
    // This avoids needing unsafe code.
    // TODO: Double check this optimization.
    let count = values.len();
    Ok(values
        .iter()
        .flat_map(|v| (*v).into())
        .collect::<Vec<f32>>()
        .into_pyarray_bound(py)
        .reshape((count, N))
        .unwrap()
        .into())
}

fn pyarray_vectors<const N: usize, T>(py: Python, values: &PyObject) -> PyResult<Vec<T>>
where
    T: From<[f32; N]>,
{
    let values: Vec<[f32; N]> = values.extract(py)?;
    Ok(values.into_iter().map(Into::into).collect())
}
