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
    NormalMapZ,
    MonochromeX,
    MonochromeY,
    MonochromeZ,
    Negate,
    FurInstanceAlpha,
    Float,
    Int,
    Uint,
    Truncate,
    FloatBitsToInt,
    IntBitsToFloat,
    UintBitsToFloat,
    InverseSqrt,
    Not,
    LeftShift,
    RightShift,
    PartialDerivativeX,
    PartialDerivativeY,
    Exp2,
    Log2,
    Sin,
    Cos
);

python_enum!(
    OperationXyz,
    xc3_model::shader_database::OperationXyz,
    Unk,
    Mix,
    Mul,
    Div,
    Add,
    Sub,
    Fma,
    MulRatio,
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
    Reflect,
    Floor,
    Select,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Monochrome,
    Negate,
    Float,
    Int,
    Uint,
    Truncate,
    FloatBitsToInt,
    IntBitsToFloat,
    UintBitsToFloat,
    InverseSqrt,
    Not,
    LeftShift,
    RightShift,
    Exp2,
    Log2,
    Sin,
    Cos
);

python_enum!(
    ChannelXyz,
    xc3_model::shader_database::ChannelXyz,
    Xyz,
    X,
    Y,
    Z,
    W
);

#[pymodule]
pub mod shader_database {

    use crate::py_exception;
    use map_py::{MapPy, TypedDict, TypedList};
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::Operation;

    #[pyclass(from_py_object)]
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

    #[pyclass(get_all, set_all, from_py_object)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::ShaderProgram)]
    pub struct ShaderProgram {
        pub output_dependencies: TypedDict<String, usize>,
        pub outline_width: Option<Value>,
        pub normal_intensity: Option<usize>,
        pub val_inf_intensity: Option<usize>,
        pub exprs: TypedList<OutputExpr>,
        pub output_dependencies_xyz: TypedDict<String, usize>,
        pub exprs_xyz: TypedList<OutputExprXyz>,
    }

    #[pyclass(from_py_object)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::OutputExpr)]
    pub struct OutputExpr(pub xc3_model::shader_database::OutputExpr);

    #[pyclass(get_all, set_all, from_py_object)]
    #[derive(Debug, Clone)]
    pub struct OutputExprFunc {
        pub op: Operation,
        pub args: Vec<usize>,
    }

    #[pyclass(from_py_object)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::Value)]
    pub struct Value(pub xc3_model::shader_database::Value);

    #[pyclass(get_all, set_all, from_py_object)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::Parameter)]
    pub struct Parameter {
        pub name: String,
        pub field: String,
        pub index: Option<usize>,
        pub channel: Option<char>,
    }

    #[pymethods]
    impl Parameter {
        #[new]
        fn new(name: String, field: String, index: Option<usize>, channel: Option<char>) -> Self {
            Self {
                name,
                field,
                index,
                channel,
            }
        }
    }

    #[pyclass(get_all, set_all, from_py_object)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::Texture)]
    pub struct Texture {
        pub name: String,
        pub channel: Option<char>,
        pub texcoords: Vec<usize>,
    }

    #[pyclass(get_all, set_all, from_py_object)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::Attribute)]
    pub struct Attribute {
        pub name: String,
        pub channel: Option<char>,
    }

    #[pymethods]
    impl OutputExpr {
        pub fn value(&self) -> Option<Value> {
            match &self.0 {
                xc3_model::shader_database::OutputExpr::Value(v) => Some(Value(v.clone())),
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
    impl Value {
        pub fn int(&self) -> Option<i32> {
            match &self.0 {
                xc3_model::shader_database::Value::Int(i) => Some(*i),
                _ => None,
            }
        }

        pub fn float(&self) -> Option<f32> {
            match &self.0 {
                xc3_model::shader_database::Value::Float(c) => Some(c.0),
                _ => None,
            }
        }

        pub fn parameter(&self, py: Python) -> PyResult<Option<Parameter>> {
            match &self.0 {
                xc3_model::shader_database::Value::Parameter(b) => b.clone().map_py(py).map(Some),
                _ => Ok(None),
            }
        }

        pub fn texture(&self, py: Python) -> PyResult<Option<Texture>> {
            match &self.0 {
                xc3_model::shader_database::Value::Texture(t) => t.clone().map_py(py).map(Some),
                _ => Ok(None),
            }
        }

        pub fn attribute(&self, py: Python) -> PyResult<Option<Attribute>> {
            match &self.0 {
                xc3_model::shader_database::Value::Attribute(a) => a.clone().map_py(py).map(Some),
                _ => Ok(None),
            }
        }
    }

    #[pyclass(from_py_object)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::OutputExprXyz)]
    pub struct OutputExprXyz(pub xc3_model::shader_database::OutputExprXyz);

    #[pyclass(from_py_object)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::shader_database::ValueXyz)]
    pub struct ValueXyz(pub xc3_model::shader_database::ValueXyz);

    #[pymodule_export]
    use super::OperationXyz;

    #[pymodule_export]
    use super::ChannelXyz;

    #[pyclass(get_all, set_all, from_py_object)]
    #[derive(Debug, Clone)]
    pub struct OutputExprFuncXyz {
        pub op: OperationXyz,
        pub args: Vec<usize>,
        pub channel: Option<ChannelXyz>,
    }

    #[pyclass(get_all, set_all, from_py_object)]
    #[derive(Debug, Clone)]
    pub struct TextureXyz {
        pub name: String,
        pub channel: Option<ChannelXyz>,
        pub texcoords: Vec<usize>,
    }

    #[pyclass(get_all, set_all, from_py_object)]
    #[derive(Debug, Clone)]
    pub struct AttributeXyz {
        pub name: String,
        pub channel: Option<ChannelXyz>,
    }

    #[pyclass(get_all, set_all, from_py_object)]
    #[derive(Debug, Clone)]
    pub struct ParameterXyz {
        pub name: String,
        pub field: String,
        pub index: Option<usize>,
        pub channel: Option<ChannelXyz>,
    }

    #[pymethods]
    impl OutputExprXyz {
        pub fn func(&self) -> Option<OutputExprFuncXyz> {
            match &self.0 {
                xc3_model::shader_database::OutputExprXyz::Func { op, args, channel } => {
                    Some(OutputExprFuncXyz {
                        op: (*op).into(),
                        args: args.clone(),
                        channel: channel.map(Into::into),
                    })
                }
                _ => None,
            }
        }

        pub fn value(&self) -> Option<ValueXyz> {
            match &self.0 {
                xc3_model::shader_database::OutputExprXyz::Value(v) => Some(ValueXyz(v.clone())),
                _ => None,
            }
        }
    }

    #[pymethods]
    impl ValueXyz {
        pub fn texture(&self) -> Option<TextureXyz> {
            match &self.0 {
                xc3_model::shader_database::ValueXyz::Texture(t) => Some(TextureXyz {
                    name: t.name.to_string(),
                    channel: t.channel.map(Into::into),
                    texcoords: t.texcoords.clone(),
                }),
                _ => None,
            }
        }

        pub fn float(&self) -> Option<(f32, f32, f32)> {
            match self.0 {
                xc3_model::shader_database::ValueXyz::Float(f) => Some((f[0].0, f[1].0, f[2].0)),
                _ => None,
            }
        }

        pub fn attribute(&self) -> Option<AttributeXyz> {
            match self.0.clone() {
                xc3_model::shader_database::ValueXyz::Attribute(a) => Some(AttributeXyz {
                    name: a.name.to_string(),
                    channel: a.channel.map(Into::into),
                }),
                _ => None,
            }
        }

        pub fn parameter(&self) -> Option<ParameterXyz> {
            match self.0.clone() {
                xc3_model::shader_database::ValueXyz::Parameter(p) => Some(ParameterXyz {
                    name: p.name.to_string(),
                    field: p.field.to_string(),
                    index: p.index,
                    channel: p.channel.map(Into::into),
                }),
                _ => None,
            }
        }
    }
}
