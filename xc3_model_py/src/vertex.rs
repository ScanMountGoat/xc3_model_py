use pyo3::{prelude::*, types::PyList};

use crate::{map_py::MapPy, skinning::Weights, uvec2s_pyarray, uvec4_pyarray};

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct ModelBuffers {
    pub vertex_buffers: Py<PyList>,
    pub index_buffers: Py<PyList>,
    pub weights: Option<Py<Weights>>,
    // TODO: add missing fields and derive conversions
}

#[pymethods]
impl ModelBuffers {
    #[new]
    pub fn new(
        vertex_buffers: Py<PyList>,
        index_buffers: Py<PyList>,
        weights: Option<Py<Weights>>,
    ) -> Self {
        Self {
            vertex_buffers,
            index_buffers,
            weights,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::vertex::VertexBuffer)]
pub struct VertexBuffer {
    pub attributes: Py<PyList>,
    pub morph_blend_target: Py<PyList>,
    pub morph_targets: Py<PyList>,
    pub outline_buffer_index: Option<usize>,
}

#[pymethods]
impl VertexBuffer {
    #[new]
    fn new(
        attributes: Py<PyList>,
        morph_blend_target: Py<PyList>,
        morph_targets: Py<PyList>,
        outline_buffer_index: Option<usize>,
    ) -> Self {
        Self {
            attributes,
            morph_blend_target,
            morph_targets,
            outline_buffer_index,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::vertex::IndexBuffer)]
pub struct IndexBuffer {
    pub indices: PyObject,
}

#[pymethods]
impl IndexBuffer {
    #[new]
    fn new(indices: PyObject) -> Self {
        Self { indices }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct AttributeData {
    pub attribute_type: AttributeType,
    // numpy.ndarray with vertex count many rows
    pub data: PyObject,
}

#[pymethods]
impl AttributeData {
    #[new]
    fn new(attribute_type: AttributeType, data: PyObject) -> Self {
        Self {
            attribute_type,
            data,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum AttributeType {
    Position,
    Normal,
    Tangent,
    TexCoord0,
    TexCoord1,
    TexCoord2,
    TexCoord3,
    TexCoord4,
    TexCoord5,
    TexCoord6,
    TexCoord7,
    TexCoord8,
    VertexColor,
    Blend,
    WeightIndex,
    Position2,
    Normal4,
    OldPosition,
    Tangent2,
    SkinWeights,
    SkinWeights2,
    BoneIndices,
    BoneIndices2,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::vertex::MorphTarget)]
pub struct MorphTarget {
    pub morph_controller_index: usize,
    // N x 3 numpy.ndarray
    pub position_deltas: PyObject,
    // N x 4 numpy.ndarray
    pub normals: PyObject,
    // N x 4 numpy.ndarray
    pub tangents: PyObject,
    pub vertex_indices: PyObject,
}

#[pymethods]
impl MorphTarget {
    #[new]
    fn new(
        morph_controller_index: usize,
        position_deltas: PyObject,
        normals: PyObject,
        tangents: PyObject,
        vertex_indices: PyObject,
    ) -> Self {
        Self {
            morph_controller_index,
            position_deltas,
            normals,
            tangents,
            vertex_indices,
        }
    }
}

pub fn vertex(py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "vertex")?;

    m.add_class::<ModelBuffers>()?;
    m.add_class::<VertexBuffer>()?;
    m.add_class::<IndexBuffer>()?;
    m.add_class::<AttributeData>()?;
    m.add_class::<AttributeType>()?;
    m.add_class::<MorphTarget>()?;

    module.add_submodule(&m)?;
    Ok(())
}

impl MapPy<xc3_model::vertex::ModelBuffers> for ModelBuffers {
    fn map_py(&self, py: Python) -> PyResult<xc3_model::vertex::ModelBuffers> {
        Ok(xc3_model::vertex::ModelBuffers {
            vertex_buffers: self.vertex_buffers.map_py(py)?,
            // TODO: Fill in all fields
            outline_buffers: Vec::new(),
            index_buffers: self.index_buffers.map_py(py)?,
            unk_buffers: Vec::new(),
            weights: self
                .weights
                .as_ref()
                .map(|w| w.extract::<Weights>(py)?.map_py(py))
                .transpose()?,
        })
    }
}

impl MapPy<ModelBuffers> for xc3_model::vertex::ModelBuffers {
    fn map_py(&self, py: Python) -> PyResult<ModelBuffers> {
        Ok(ModelBuffers {
            vertex_buffers: self.vertex_buffers.map_py(py)?,
            index_buffers: self.index_buffers.map_py(py)?,
            weights: match self.weights.as_ref() {
                Some(w) => Some(Py::new(py, w.map_py(py)?)?),
                None => None,
            },
        })
    }
}

// Map from Python lists to Vec<T>
impl crate::MapPy<Vec<xc3_model::vertex::ModelBuffers>> for Py<PyList> {
    fn map_py(&self, py: Python) -> PyResult<Vec<xc3_model::vertex::ModelBuffers>> {
        self.extract::<'_, '_, Vec<ModelBuffers>>(py)?
            .iter()
            .map(|v| v.map_py(py))
            .collect::<Result<Vec<_>, _>>()
    }
}

// Map from Vec<T> to Python lists
impl crate::MapPy<Py<PyList>> for Vec<xc3_model::vertex::ModelBuffers> {
    fn map_py(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new_bound(
            py,
            self.iter()
                .map(|v| {
                    let v2: ModelBuffers = v.map_py(py)?;
                    Ok(v2.into_py(py))
                })
                .collect::<PyResult<Vec<_>>>()?,
        )
        .into())
    }
}

// Map to and from Py<T>
impl MapPy<Py<ModelBuffers>> for xc3_model::vertex::ModelBuffers {
    fn map_py(&self, py: Python) -> PyResult<Py<ModelBuffers>> {
        let value: ModelBuffers = self.map_py(py)?;
        Py::new(py, value)
    }
}

impl MapPy<xc3_model::vertex::ModelBuffers> for Py<ModelBuffers> {
    fn map_py(&self, py: Python) -> PyResult<xc3_model::vertex::ModelBuffers> {
        self.extract::<ModelBuffers>(py)?.map_py(py)
    }
}

impl MapPy<AttributeData> for xc3_model::vertex::AttributeData {
    fn map_py(&self, py: Python) -> PyResult<AttributeData> {
        match self {
            xc3_model::vertex::AttributeData::Position(values) => Ok(AttributeData {
                attribute_type: AttributeType::Position,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::Normal(values) => Ok(AttributeData {
                attribute_type: AttributeType::Normal,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::Tangent(values) => Ok(AttributeData {
                attribute_type: AttributeType::Tangent,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::TexCoord0(values) => Ok(AttributeData {
                attribute_type: AttributeType::TexCoord0,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::TexCoord1(values) => Ok(AttributeData {
                attribute_type: AttributeType::TexCoord1,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::TexCoord2(values) => Ok(AttributeData {
                attribute_type: AttributeType::TexCoord2,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::TexCoord3(values) => Ok(AttributeData {
                attribute_type: AttributeType::TexCoord3,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::TexCoord4(values) => Ok(AttributeData {
                attribute_type: AttributeType::TexCoord4,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::TexCoord5(values) => Ok(AttributeData {
                attribute_type: AttributeType::TexCoord5,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::TexCoord6(values) => Ok(AttributeData {
                attribute_type: AttributeType::TexCoord6,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::TexCoord7(values) => Ok(AttributeData {
                attribute_type: AttributeType::TexCoord7,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::TexCoord8(values) => Ok(AttributeData {
                attribute_type: AttributeType::TexCoord8,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::VertexColor(values) => Ok(AttributeData {
                attribute_type: AttributeType::VertexColor,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::Blend(values) => Ok(AttributeData {
                attribute_type: AttributeType::Blend,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::WeightIndex(values) => Ok(AttributeData {
                attribute_type: AttributeType::WeightIndex,
                data: uvec2s_pyarray(py, values),
            }),
            xc3_model::vertex::AttributeData::Position2(values) => Ok(AttributeData {
                attribute_type: AttributeType::Position2,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::Normal4(values) => Ok(AttributeData {
                attribute_type: AttributeType::Normal4,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::OldPosition(values) => Ok(AttributeData {
                attribute_type: AttributeType::OldPosition,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::Tangent2(values) => Ok(AttributeData {
                attribute_type: AttributeType::Tangent2,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::SkinWeights(values) => Ok(AttributeData {
                attribute_type: AttributeType::SkinWeights,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::SkinWeights2(values) => Ok(AttributeData {
                attribute_type: AttributeType::SkinWeights2,
                data: values.map_py(py)?,
            }),
            xc3_model::vertex::AttributeData::BoneIndices(values) => Ok(AttributeData {
                attribute_type: AttributeType::BoneIndices,
                data: uvec4_pyarray(py, values),
            }),
            xc3_model::vertex::AttributeData::BoneIndices2(values) => Ok(AttributeData {
                attribute_type: AttributeType::BoneIndices2,
                data: uvec4_pyarray(py, values),
            }),
        }
    }
}

impl MapPy<xc3_model::vertex::AttributeData> for AttributeData {
    fn map_py(&self, py: Python) -> PyResult<xc3_model::vertex::AttributeData> {
        use xc3_model::vertex::AttributeData as AttrRs;

        match self.attribute_type {
            AttributeType::Position => Ok(AttrRs::Position(self.data.map_py(py)?)),
            AttributeType::Normal => Ok(AttrRs::Normal(self.data.map_py(py)?)),
            AttributeType::Tangent => Ok(AttrRs::Tangent(self.data.map_py(py)?)),
            AttributeType::TexCoord0 => Ok(AttrRs::TexCoord0(self.data.map_py(py)?)),
            AttributeType::TexCoord1 => Ok(AttrRs::TexCoord1(self.data.map_py(py)?)),
            AttributeType::TexCoord2 => Ok(AttrRs::TexCoord2(self.data.map_py(py)?)),
            AttributeType::TexCoord3 => Ok(AttrRs::TexCoord3(self.data.map_py(py)?)),
            AttributeType::TexCoord4 => Ok(AttrRs::TexCoord4(self.data.map_py(py)?)),
            AttributeType::TexCoord5 => Ok(AttrRs::TexCoord5(self.data.map_py(py)?)),
            AttributeType::TexCoord6 => Ok(AttrRs::TexCoord6(self.data.map_py(py)?)),
            AttributeType::TexCoord7 => Ok(AttrRs::TexCoord7(self.data.map_py(py)?)),
            AttributeType::TexCoord8 => Ok(AttrRs::TexCoord8(self.data.map_py(py)?)),
            AttributeType::VertexColor => Ok(AttrRs::VertexColor(self.data.map_py(py)?)),
            AttributeType::Blend => Ok(AttrRs::Blend(self.data.map_py(py)?)),
            AttributeType::WeightIndex => Ok(AttrRs::WeightIndex(self.data.extract(py)?)),
            AttributeType::Position2 => Ok(AttrRs::Position2(self.data.map_py(py)?)),
            AttributeType::Normal4 => Ok(AttrRs::Normal4(self.data.map_py(py)?)),
            AttributeType::OldPosition => Ok(AttrRs::OldPosition(self.data.map_py(py)?)),
            AttributeType::Tangent2 => Ok(AttrRs::Tangent2(self.data.map_py(py)?)),
            AttributeType::SkinWeights => Ok(AttrRs::SkinWeights(self.data.map_py(py)?)),
            AttributeType::SkinWeights2 => Ok(AttrRs::SkinWeights2(self.data.map_py(py)?)),
            AttributeType::BoneIndices => Ok(AttrRs::BoneIndices(self.data.extract(py)?)),
            AttributeType::BoneIndices2 => Ok(AttrRs::BoneIndices2(self.data.extract(py)?)),
        }
    }
}

// Map from Python lists to Vec<T>
impl crate::MapPy<Vec<xc3_model::vertex::AttributeData>> for Py<PyList> {
    fn map_py(&self, py: Python) -> PyResult<Vec<xc3_model::vertex::AttributeData>> {
        self.extract::<'_, '_, Vec<AttributeData>>(py)?
            .iter()
            .map(|v| v.map_py(py))
            .collect::<Result<Vec<_>, _>>()
    }
}

// Map from Vec<T> to Python lists
impl crate::MapPy<Py<PyList>> for Vec<xc3_model::vertex::AttributeData> {
    fn map_py(&self, py: Python) -> PyResult<Py<PyList>> {
        Ok(PyList::new_bound(
            py,
            self.iter()
                .map(|v| Ok(v.map_py(py)?.into_py(py)))
                .collect::<PyResult<Vec<_>>>()?,
        )
        .into())
    }
}
