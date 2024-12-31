use glam::{Mat4, Vec2, Vec3, Vec4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::{prelude::*, types::PyList};
use smol_str::SmolStr;
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

map_py_impl!(
    char,
    bool,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    String,
    Vec<u8>,
    Vec<f32>,
    Vec<(u16, u16)>,
    Vec<[f32; 4]>,
    Vec<[f32; 8]>
);

#[macro_export]
macro_rules! map_py_into_impl {
    ($t:ty,$u:ty) => {
        impl MapPy<$u> for $t {
            fn map_py(&self, _py: Python) -> PyResult<$u> {
                Ok((*self).into())
            }
        }

        impl MapPy<$t> for $u {
            fn map_py(&self, _py: Python) -> PyResult<$t> {
                Ok((*self).into())
            }
        }
    };
}

map_py_into_impl!(Vec3, [f32; 3]);

#[macro_export]
macro_rules! map_py_pyobject_ndarray_impl {
    ($($t:ty),*) => {
        $(
            impl MapPy<PyObject> for Vec<$t> {
                fn map_py(&self, py: Python) -> PyResult<PyObject> {
                    Ok(self.to_pyarray(py).into_any().into())
                }
            }

            impl MapPy<Vec<$t>> for PyObject {
                fn map_py(&self, py: Python) -> PyResult<Vec<$t>> {
                    self.extract(py)
                }
            }

            impl MapPy<Py<PyArray1<$t>>> for Vec<$t> {
                fn map_py(&self, py: Python) -> PyResult<Py<PyArray1<$t>>> {
                    Ok(self.to_pyarray(py).into())
                }
            }

            impl MapPy<Vec<$t>> for Py<PyArray1<$t>> {
                fn map_py(&self, py: Python) -> PyResult<Vec<$t>> {
                    let array = self.as_any().downcast_bound::<PyArray1<$t>>(py)?;
                    Ok(array.readonly().as_slice()?.to_vec())
                }
            }
        )*
    }
}

map_py_pyobject_ndarray_impl!(u16, u32, f32);

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
        PyList::new(py, self).map(Into::into)
    }
}

impl MapPy<PyObject> for Vec<Vec2> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        vectors_pyarray(py, self)
    }
}

impl MapPy<Vec<Vec2>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<Vec2>> {
        pyarray_vec2s(py, self)
    }
}

impl MapPy<PyObject> for Vec<Vec3> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        vectors_pyarray(py, self)
    }
}

impl MapPy<Vec<Vec3>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<Vec3>> {
        pyarray_vec3s(py, self)
    }
}

impl MapPy<PyObject> for Vec<Vec4> {
    fn map_py(&self, py: Python) -> PyResult<PyObject> {
        vectors_pyarray(py, self)
    }
}

impl MapPy<Vec<Vec4>> for PyObject {
    fn map_py(&self, py: Python) -> PyResult<Vec<Vec4>> {
        pyarray_vec4s(py, self)
    }
}

macro_rules! map_py_vecn_ndarray_impl {
    ($t:ty,$n:expr) => {
        impl MapPy<Py<PyArray2<f32>>> for Vec<$t> {
            fn map_py(&self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
                // This flatten will be optimized in Release mode.
                // This avoids needing unsafe code.
                // TODO: Double check this optimization.
                // TODO: faster to use bytemuck?
                let count = self.len();
                Ok(self
                    .iter()
                    .flat_map(|v| v.to_array())
                    .collect::<Vec<f32>>()
                    .into_pyarray(py)
                    .reshape((count, $n))
                    .unwrap()
                    .into())
            }
        }

        impl MapPy<Vec<$t>> for Py<PyArray2<f32>> {
            fn map_py(&self, py: Python) -> PyResult<Vec<$t>> {
                let array = self.as_any().downcast_bound::<PyArray2<f32>>(py)?;
                Ok(array
                    .readonly()
                    .as_array()
                    .rows()
                    .into_iter()
                    .map(|r| <$t>::from_slice(r.as_slice().unwrap()))
                    .collect())
            }
        }
    };
}
map_py_vecn_ndarray_impl!(Vec2, 2);
map_py_vecn_ndarray_impl!(Vec3, 3);
map_py_vecn_ndarray_impl!(Vec4, 4);

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

impl MapPy<Py<PyArray2<u8>>> for Vec<[u8; 4]> {
    fn map_py(&self, py: Python) -> PyResult<Py<PyArray2<u8>>> {
        // This flatten will be optimized in Release mode.
        // This avoids needing unsafe code.
        let count = self.len();
        Ok(self
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<u8>>()
            .into_pyarray(py)
            .reshape((count, 4))
            .unwrap()
            .into())
    }
}

impl MapPy<Vec<[u8; 4]>> for Py<PyArray2<u8>> {
    fn map_py(&self, py: Python) -> PyResult<Vec<[u8; 4]>> {
        let array = self.as_any().downcast_bound::<PyArray2<u8>>(py)?;
        Ok(array
            .readonly()
            .as_array()
            .rows()
            .into_iter()
            .map(|r| r.as_slice().unwrap().try_into().unwrap())
            .collect())
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
        .into_pyarray(py)
        .reshape((count, N))
        .unwrap()
        .into_any()
        .into())
}

fn pyarray_vec2s(py: Python, values: &PyObject) -> PyResult<Vec<Vec2>> {
    let array = values.downcast_bound::<PyArray2<f32>>(py)?;
    Ok(array
        .readonly()
        .as_array()
        .rows()
        .into_iter()
        .map(|r| Vec2::from_slice(r.as_slice().unwrap()))
        .collect())
}

fn pyarray_vec3s(py: Python, values: &PyObject) -> PyResult<Vec<Vec3>> {
    let array = values.downcast_bound::<PyArray2<f32>>(py)?;
    Ok(array
        .readonly()
        .as_array()
        .rows()
        .into_iter()
        .map(|r| Vec3::from_slice(r.as_slice().unwrap()))
        .collect())
}

fn pyarray_vec4s(py: Python, values: &PyObject) -> PyResult<Vec<Vec4>> {
    let array = values.downcast_bound::<PyArray2<f32>>(py)?;
    Ok(array
        .readonly()
        .as_array()
        .rows()
        .into_iter()
        .map(|r| Vec4::from_slice(r.as_slice().unwrap()))
        .collect())
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

// TODO: blanket impl using From/Into?
impl MapPy<SmolStr> for String {
    fn map_py(&self, _py: Python) -> PyResult<SmolStr> {
        Ok(self.into())
    }
}

impl MapPy<String> for SmolStr {
    fn map_py(&self, _py: Python) -> PyResult<String> {
        Ok(self.to_string())
    }
}

// TODO: const generics?
impl<T, U, const N: usize> MapPy<[U; N]> for [T; N]
where
    T: MapPy<U>,
{
    fn map_py(&self, py: Python) -> PyResult<[U; N]> {
        // TODO: avoid unwrap
        Ok(std::array::from_fn(|i| self[i].map_py(py).unwrap()))
    }
}

#[macro_export]
macro_rules! map_py_wrapper_impl {
    ($rs:ty, $py:path) => {
        impl MapPy<$rs> for $py {
            fn map_py(&self, _py: Python) -> PyResult<$rs> {
                Ok(self.0.clone())
            }
        }

        impl MapPy<$py> for $rs {
            fn map_py(&self, _py: Python) -> PyResult<$py> {
                Ok($py(self.clone()))
            }
        }

        // Map to and from Py<T>
        impl MapPy<Py<$py>> for $rs {
            fn map_py(&self, py: Python) -> PyResult<Py<$py>> {
                let value: $py = self.map_py(py)?;
                Py::new(py, value)
            }
        }

        impl MapPy<$rs> for Py<$py> {
            fn map_py(&self, py: Python) -> PyResult<$rs> {
                self.extract::<$py>(py)?.map_py(py)
            }
        }

        // Map from Python lists to Vec<T>
        impl MapPy<Vec<$rs>> for Py<pyo3::types::PyList> {
            fn map_py(&self, py: Python) -> PyResult<Vec<$rs>> {
                self.extract::<'_, '_, Vec<$py>>(py)?
                    .iter()
                    .map(|v| v.map_py(py))
                    .collect::<Result<Vec<_>, _>>()
            }
        }

        // Map from Vec<T> to Python lists
        impl MapPy<Py<pyo3::types::PyList>> for Vec<$rs> {
            fn map_py(&self, py: Python) -> PyResult<Py<pyo3::types::PyList>> {
                pyo3::types::PyList::new(
                    py,
                    self.into_iter()
                        .map(|v| {
                            let v2: $py = v.map_py(py)?;
                            v2.into_pyobject(py)
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                )
                .map(Into::into)
            }
        }
    };
}
