use crate::python_enum;
use pyo3::prelude::*;

python_enum!(
    Operation,
    xc3_model::shader_database::Operation,
    Unk,
    Mix,
    Mul,
    Div,
    Add,
    Sub,
    Fma,
    MulRatio,
    AddNormalX,
    AddNormalY,
    Overlay,
    Overlay2,
    OverlayRatio,
    Power,
    Min,
    Max,
    Clamp,
    Abs,
    Fresnel,
    Sqrt,
    TexMatrix,
    TexParallaxX,
    TexParallaxY,
    ReflectX,
    ReflectY,
    ReflectZ,
    Floor,
    Select,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Dot4,
    NormalMapX,
    NormalMapY,
    NormalMapZ
);

#[pymodule]
pub mod shader_database {

    use crate::py_exception;
    use map_py::{MapPy, TypedList};
    use pyo3::{prelude::*, types::PyDict};

    #[pymodule_export]
    use super::Operation;

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
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::ShaderProgram)]
    pub struct ShaderProgram {
        pub output_dependencies: Py<PyDict>,
        pub outline_width: Option<Dependency>,
        pub normal_intensity: Option<usize>,
        pub exprs: TypedList<OutputExpr>,
    }

    #[pyclass]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::OutputExpr)]
    pub struct OutputExpr(xc3_model::shader_database::OutputExpr);

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct OutputExprFunc {
        pub op: Operation,
        pub args: Vec<usize>,
    }

    #[pyclass]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::Dependency)]
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
        pub texcoords: Vec<usize>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::AttributeDependency)]
    pub struct AttributeDependency {
        pub name: String,
        pub channel: Option<char>,
    }

    #[pymethods]
    impl OutputExpr {
        pub fn value(&self) -> Option<Dependency> {
            match &self.0 {
                xc3_model::shader_database::OutputExpr::Value(v) => Some(Dependency(v.clone())),
                _ => None,
            }
        }

        pub fn func(&self) -> Option<OutputExprFunc> {
            match &self.0 {
                xc3_model::shader_database::OutputExpr::Func { op, args } => Some(OutputExprFunc {
                    op: (*op).into(),
                    args: args.clone(),
                }),
                _ => None,
            }
        }
    }

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
                xc3_model::shader_database::Dependency::Buffer(b) => b.clone().map_py(py).map(Some),
                _ => Ok(None),
            }
        }

        pub fn texture(&self, py: Python) -> PyResult<Option<TextureDependency>> {
            match &self.0 {
                xc3_model::shader_database::Dependency::Texture(t) => {
                    t.clone().map_py(py).map(Some)
                }
                _ => Ok(None),
            }
        }

        pub fn attribute(&self, py: Python) -> PyResult<Option<AttributeDependency>> {
            match &self.0 {
                xc3_model::shader_database::Dependency::Attribute(a) => {
                    a.clone().map_py(py).map(Some)
                }
                _ => Ok(None),
            }
        }
    }
}
