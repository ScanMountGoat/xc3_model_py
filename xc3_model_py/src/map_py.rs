use glam::{Mat4, Vec2, Vec3, Vec4};
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::{prelude::*, types::PyList};
pub use xc3_model_py_derive::MapPy;

use crate::{
    mat4_to_pyarray, pyarray_to_mat4, pyarray_to_mat4s, transforms_pyarray, uvec4_pyarray,
};

// Define a mapping between types.
// This allows for deriving the Python <-> Rust conversion.
// The derive macro is mainly to automate mapping field names.
pub trait MapPy<T> {
    // TODO: take self by value to improve performance?
    fn map_py(&self, py: Python) -> PyResult<T>;
}

// TODO: can this be a blanket impl?
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

impl MapPy<[f32; 3]> for Vec3 {
    fn map_py(&self, _py: Python) -> PyResult<[f32; 3]> {
        Ok(self.to_array())
    }
}

impl MapPy<Vec3> for [f32; 3] {
    fn map_py(&self, _py: Python) -> PyResult<Vec3> {
        Ok((*self).into())
    }
}

// TODO: macro for this
impl MapPy<PyObject> for Vec<u32> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        // TODO: avoid clone?
        Ok(self.clone().into_pyarray_bound(py).into())
    }
}

impl MapPy<Vec<u32>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<u32>> {
        self.extract(py)
    }
}

impl MapPy<PyObject> for Vec<u16> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        // TODO: avoid clone?
        Ok(self.clone().into_pyarray_bound(py).into())
    }
}

impl MapPy<Vec<u16>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<u16>> {
        self.extract(py)
    }
}

impl<T, U> MapPy<Option<U>> for Option<T>
where
    T: MapPy<U>,
{
    fn map_py(&self, py: Python) -> PyResult<Option<U>> {
        self.as_ref().map(|v| v.map_py(py)).transpose()
    }
}

// TODO: how to implement for Py<T>?

// TODO: Derive for each type to avoid overlapping definitions with numpy and PyObject?
impl MapPy<Vec<String>> for Py<PyList> {
    fn map_py(&self, py: Python) -> PyResult<Vec<String>> {
        self.extract(py)
    }
}

impl MapPy<Py<PyList>> for Vec<String> {
    fn map_py(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new_bound(py, self).into())
    }
}

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

impl MapPy<PyObject> for Vec<[u8; 4]> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        Ok(uvec4_pyarray(py, self))
    }
}

impl MapPy<Vec<[u8; 4]>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<[u8; 4]>> {
        // TODO: blanket impl for this?
        self.extract(py)
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

impl MapPy<PyObject> for Mat4 {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        Ok(mat4_to_pyarray(py, *self))
    }
}

impl MapPy<Mat4> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Mat4> {
        pyarray_to_mat4(py, self)
    }
}

impl MapPy<PyObject> for Vec<Mat4> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        Ok(transforms_pyarray(py, self))
    }
}

impl MapPy<Vec<Mat4>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<Mat4>> {
        pyarray_to_mat4s(py, self)
    }
}
