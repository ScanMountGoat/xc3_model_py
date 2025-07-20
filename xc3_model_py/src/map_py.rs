use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArray, ToPyArray};
use pyo3::{prelude::*, types::PyList};
use smol_str::SmolStr;
pub use xc3_model_py_derive::MapPy;

// Define a mapping between types.
// This allows for deriving the Python <-> Rust conversion.
// The derive macro is mainly to automate mapping field names.
pub trait MapPy<T> {
    // TODO: take self by value to improve performance?
    fn map_py(self, py: Python) -> PyResult<T>;
}

// TODO: can this be a blanket impl?
// Implement for primitive types.
macro_rules! map_py_impl {
    ($($t:ty),*) => {
        $(
            impl MapPy<$t> for $t {
                fn map_py(self, _py: Python) -> PyResult<$t> {
                    Ok(self)
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
    (u16, u16)
);

#[macro_export]
macro_rules! map_py_into_impl {
    ($t:ty,$u:ty) => {
        impl MapPy<$u> for $t {
            fn map_py(self, _py: Python) -> PyResult<$u> {
                Ok(self.into())
            }
        }

        impl MapPy<$t> for $u {
            fn map_py(self, _py: Python) -> PyResult<$t> {
                Ok(self.into())
            }
        }
    };
}

map_py_into_impl!(Vec3, [f32; 3]);

impl MapPy<Quat> for [f32; 4] {
    fn map_py(self, _py: Python) -> PyResult<Quat> {
        Ok(Quat::from_array(self))
    }
}

impl MapPy<[f32; 4]> for Quat {
    fn map_py(self, _py: Python) -> PyResult<[f32; 4]> {
        Ok(self.to_array())
    }
}

#[macro_export]
macro_rules! map_py_pyobject_ndarray_impl {
    ($($t:ty),*) => {
        $(
            // 1D arrays
            impl MapPy<Py<PyArray1<$t>>> for Vec<$t> {
                fn map_py(self, py: Python) -> PyResult<Py<PyArray1<$t>>> {
                    Ok(self.to_pyarray(py).into())
                }
            }

            impl MapPy<Vec<$t>> for Py<PyArray1<$t>> {
                fn map_py(self, py: Python) -> PyResult<Vec<$t>> {
                    let array = self.as_any().downcast_bound::<PyArray1<$t>>(py)?;
                    Ok(array.readonly().as_slice()?.to_vec())
                }
            }

            // 1D untyped arrays
            impl MapPy<Py<PyUntypedArray>> for Vec<$t> {
                fn map_py(self, py: Python) -> PyResult<Py<PyUntypedArray>> {
                    let arr: Py<PyArray1<$t>> = self.map_py(py)?;
                    Ok(arr.bind(py).as_untyped().clone().unbind())
                }
            }

            impl MapPy<Vec<$t>> for Py<PyUntypedArray> {
                fn map_py(self, py: Python) -> PyResult<Vec<$t>> {
                    let arr = self.bind(py).downcast::<PyArray1<$t>>()?;
                    arr.as_unbound().clone().map_py(py)
                }
            }

            // 2D arrays
            impl<const N: usize> MapPy<Py<PyArray2<$t>>> for Vec<[$t; N]> {
                fn map_py(self, py: Python) -> PyResult<Py<PyArray2<$t>>> {
                    // This flatten will be optimized in Release mode.
                    // This avoids needing unsafe code.
                    let count = self.len();
                    Ok(self
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<Vec<$t>>()
                        .into_pyarray(py)
                        .reshape((count, N))
                        .unwrap()
                        .into())
                }
            }

            impl<const N: usize> MapPy<Vec<[$t; N]>> for Py<PyArray2<$t>> {
                fn map_py(self, py: Python) -> PyResult<Vec<[$t; N]>> {
                    let array = self.as_any().downcast_bound::<PyArray2<$t>>(py)?;
                    Ok(array
                        .readonly()
                        .as_array()
                        .rows()
                        .into_iter()
                        .map(|r| r.as_slice().unwrap().try_into().unwrap())
                        .collect())
                }
            }

            // 2D untyped arrrays
            impl<const N: usize> MapPy<Py<PyUntypedArray>> for Vec<[$t; N]> {
                fn map_py(self, py: Python) -> PyResult<Py<PyUntypedArray>> {
                    let arr: Py<PyArray2<$t>> = self.map_py(py)?;
                    Ok(arr.bind(py).as_untyped().clone().unbind())
                }
            }

            impl<const N: usize> MapPy<Vec<[$t; N]>> for Py<PyUntypedArray> {
                fn map_py(self, py: Python) -> PyResult<Vec<[$t; N]>> {
                    let arr = self.bind(py).downcast::<PyArray2<$t>>()?;
                    arr.as_unbound().clone().map_py(py)
                }
            }
        )*
    }
}

map_py_pyobject_ndarray_impl!(u8, u16, u32, f32);

impl<T, U> MapPy<Option<U>> for Option<T>
where
    T: MapPy<U>,
{
    fn map_py(self, py: Python) -> PyResult<Option<U>> {
        self.map(|v| v.map_py(py)).transpose()
    }
}

impl<T, U> MapPy<Vec<U>> for Vec<T>
where
    T: MapPy<U>,
{
    fn map_py(self, py: Python) -> PyResult<Vec<U>> {
        self.into_iter().map(|v| v.map_py(py)).collect()
    }
}

// TODO: how to implement for Py<T>?

pub fn map_list<T, U>(list: Py<PyList>, py: Python) -> PyResult<Vec<U>>
where
    for<'a> Vec<T>: FromPyObject<'a>,
    T: MapPy<U>,
{
    list.extract::<'_, '_, Vec<T>>(py)?.map_py(py)
}

pub fn map_vec<T, U>(value: Vec<T>, py: Python) -> PyResult<Py<PyList>>
where
    T: MapPy<U>,
    for<'a> U: IntoPyObject<'a>,
    for<'a> <U as IntoPyObject<'a>>::Output: IntoPyObject<'a>,
    for<'a> <U as IntoPyObject<'a>>::Error: From<pyo3::PyErr>,
{
    PyList::new(
        py,
        value
            .into_iter()
            .map(|v| {
                let v2: U = v.map_py(py)?;
                v2.into_pyobject(py).map_err(Into::into)
            })
            .collect::<PyResult<Vec<_>>>()?,
    )
    .map(Into::into)
}

// TODO: Blanket impl without overlap?
impl MapPy<Vec<String>> for Py<PyList> {
    fn map_py(self, py: Python) -> PyResult<Vec<String>> {
        self.extract(py)
    }
}

impl MapPy<Py<PyList>> for Vec<String> {
    fn map_py(self, py: Python) -> PyResult<Py<PyList>> {
        PyList::new(py, self).map(Into::into)
    }
}

macro_rules! map_py_vecn_ndarray_impl {
    ($t:ty,$n:expr) => {
        impl MapPy<Py<PyArray2<f32>>> for Vec<$t> {
            fn map_py(self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
                // This flatten will be optimized in Release mode.
                // This avoids needing unsafe code.
                // TODO: Double check this optimization.
                // TODO: faster to use bytemuck?
                let count = self.len();
                Ok(self
                    .into_iter()
                    .flat_map(|v| v.to_array())
                    .collect::<Vec<f32>>()
                    .into_pyarray(py)
                    .reshape((count, $n))
                    .unwrap()
                    .into())
            }
        }

        impl MapPy<Vec<$t>> for Py<PyArray2<f32>> {
            fn map_py(self, py: Python) -> PyResult<Vec<$t>> {
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

        impl MapPy<Py<PyUntypedArray>> for Vec<$t> {
            fn map_py(self, py: Python) -> PyResult<Py<PyUntypedArray>> {
                let arr: Py<PyArray2<f32>> = self.map_py(py)?;
                Ok(arr.bind(py).as_untyped().clone().unbind())
            }
        }

        impl MapPy<Vec<$t>> for Py<PyUntypedArray> {
            fn map_py(self, py: Python) -> PyResult<Vec<$t>> {
                let arr = self.bind(py).downcast::<PyArray2<f32>>()?;
                arr.as_unbound().clone().map_py(py)
            }
        }
    };
}
map_py_vecn_ndarray_impl!(Vec2, 2);
map_py_vecn_ndarray_impl!(Vec3, 3);
map_py_vecn_ndarray_impl!(Vec4, 4);
map_py_vecn_ndarray_impl!(Quat, 4);

impl MapPy<Py<PyArray2<f32>>> for Mat4 {
    fn map_py(self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
        // TODO: Should this be transposed since numpy is row-major?
        Ok(self
            .to_cols_array()
            .to_pyarray(py)
            .readwrite()
            .reshape((4, 4))
            .unwrap()
            .into())
    }
}

impl MapPy<Mat4> for Py<PyArray2<f32>> {
    fn map_py(self, py: Python) -> PyResult<Mat4> {
        let array = self.as_any().downcast_bound::<PyArray2<f32>>(py)?;
        Ok(Mat4::from_cols_slice(
            array.readonly().as_array().as_slice().unwrap(),
        ))
    }
}

impl MapPy<Py<PyArray3<f32>>> for Vec<Mat4> {
    fn map_py(self, py: Python) -> PyResult<Py<PyArray3<f32>>> {
        // This flatten will be optimized in Release mode.
        // This avoids needing unsafe code.
        // TODO: transpose?
        let count = self.len();
        Ok(self
            .iter()
            .flat_map(|v| v.to_cols_array())
            .collect::<Vec<f32>>()
            .into_pyarray(py)
            .reshape((count, 4, 4))
            .unwrap()
            .into())
    }
}

impl MapPy<Vec<Mat4>> for Py<PyArray3<f32>> {
    fn map_py(self, py: Python) -> PyResult<Vec<Mat4>> {
        let array = self.as_any().downcast_bound::<PyArray3<f32>>(py)?;
        let array = array.readonly();
        let array = array.as_array();
        Ok(array
            .into_shape_with_order((array.shape()[0], 16))
            .unwrap()
            .rows()
            .into_iter()
            .map(|r| Mat4::from_cols_slice(r.as_slice().unwrap()))
            .collect())
    }
}

// TODO: blanket impl using From/Into?
impl MapPy<SmolStr> for String {
    fn map_py(self, _py: Python) -> PyResult<SmolStr> {
        Ok(self.into())
    }
}

impl MapPy<String> for SmolStr {
    fn map_py(self, _py: Python) -> PyResult<String> {
        Ok(self.to_string())
    }
}

impl<T, U, const N: usize> MapPy<[U; N]> for [T; N]
where
    T: MapPy<U>,
{
    fn map_py(self, py: Python) -> PyResult<[U; N]> {
        // TODO: avoid unwrap
        Ok(self.map(|i| i.map_py(py).unwrap()))
    }
}

#[macro_export]
macro_rules! map_py_wrapper_impl {
    ($rs:ty, $py:path) => {
        impl MapPy<$rs> for $py {
            fn map_py(self, _py: Python) -> PyResult<$rs> {
                Ok(self.0)
            }
        }

        impl MapPy<$py> for $rs {
            fn map_py(self, _py: Python) -> PyResult<$py> {
                Ok($py(self))
            }
        }

        // Map to and from Py<T>
        impl MapPy<Py<$py>> for $rs {
            fn map_py(self, py: Python) -> PyResult<Py<$py>> {
                let value: $py = self.map_py(py)?;
                Py::new(py, value)
            }
        }

        impl MapPy<$rs> for Py<$py> {
            fn map_py(self, py: Python) -> PyResult<$rs> {
                self.extract::<$py>(py)?.map_py(py)
            }
        }

        // Map from Python lists to Vec<T>
        impl MapPy<Vec<$rs>> for Py<pyo3::types::PyList> {
            fn map_py(self, py: Python) -> PyResult<Vec<$rs>> {
                self.extract::<'_, '_, Vec<$py>>(py)?
                    .into_iter()
                    .map(|v| v.map_py(py))
                    .collect::<Result<Vec<_>, _>>()
            }
        }

        // Map from Vec<T> to Python lists
        impl MapPy<Py<pyo3::types::PyList>> for Vec<$rs> {
            fn map_py(self, py: Python) -> PyResult<Py<pyo3::types::PyList>> {
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
