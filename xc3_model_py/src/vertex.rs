use numpy::IntoPyArray;
use pyo3::{prelude::*, types::PyList};

use crate::{
    map_py::MapPy,
    skinning::{weights_py, weights_rs, Weights},
    uvec2s_pyarray, uvec4_pyarray,
};

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct ModelBuffers {
    pub vertex_buffers: Py<PyList>,
    pub index_buffers: Py<PyList>,
    pub weights: Option<Py<Weights>>,
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

pub fn model_buffers_rs(
    py: Python,
    buffer: &ModelBuffers,
) -> PyResult<xc3_model::vertex::ModelBuffers> {
    Ok(xc3_model::vertex::ModelBuffers {
        vertex_buffers: buffer
            .vertex_buffers
            .extract::<'_, '_, Vec<VertexBuffer>>(py)?
            .iter()
            .map(|b| vertex_buffer_rs(py, b))
            .collect::<PyResult<Vec<_>>>()?,
        // TODO: Fill in all fields
        outline_buffers: Vec::new(),
        index_buffers: buffer
            .index_buffers
            .extract::<'_, '_, Vec<IndexBuffer>>(py)?
            .iter()
            .map(|b| {
                Ok(xc3_model::vertex::IndexBuffer {
                    indices: b.indices.extract(py)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        unk_buffers: Vec::new(),
        weights: buffer
            .weights
            .as_ref()
            .map(|w| weights_rs(py, &w.extract(py)?))
            .transpose()?,
    })
}

pub fn model_buffers_py(
    py: Python,
    buffer: xc3_model::vertex::ModelBuffers,
) -> PyResult<ModelBuffers> {
    Ok(ModelBuffers {
        vertex_buffers: vertex_buffers_py(py, buffer.vertex_buffers)?,
        index_buffers: index_buffers_py(py, buffer.index_buffers),
        weights: match buffer.weights {
            Some(w) => Some(Py::new(py, weights_py(py, w)?)?),
            None => None,
        },
    })
}

fn vertex_attributes_py(
    py: Python,
    attributes: Vec<xc3_model::vertex::AttributeData>,
) -> PyResult<Py<PyList>> {
    Ok(PyList::new_bound(
        py,
        attributes
            .into_iter()
            .map(|attribute| Ok(attribute_data_py(py, attribute)?.into_py(py)))
            .collect::<PyResult<Vec<_>>>()?,
    )
    .into())
}

fn attribute_data_py(
    py: Python,
    attribute: xc3_model::vertex::AttributeData,
) -> PyResult<AttributeData> {
    match attribute {
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
            data: uvec2s_pyarray(py, &values),
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
            data: uvec4_pyarray(py, &values),
        }),
        xc3_model::vertex::AttributeData::BoneIndices2(values) => Ok(AttributeData {
            attribute_type: AttributeType::BoneIndices2,
            data: uvec4_pyarray(py, &values),
        }),
    }
}

fn attribute_data_rs(
    py: Python,
    attribute: &AttributeData,
) -> PyResult<xc3_model::vertex::AttributeData> {
    use xc3_model::vertex::AttributeData as AttrRs;

    match attribute.attribute_type {
        AttributeType::Position => Ok(AttrRs::Position(attribute.data.map_py(py)?)),
        AttributeType::Normal => Ok(AttrRs::Normal(attribute.data.map_py(py)?)),
        AttributeType::Tangent => Ok(AttrRs::Tangent(attribute.data.map_py(py)?)),
        AttributeType::TexCoord0 => Ok(AttrRs::TexCoord0(attribute.data.map_py(py)?)),
        AttributeType::TexCoord1 => Ok(AttrRs::TexCoord1(attribute.data.map_py(py)?)),
        AttributeType::TexCoord2 => Ok(AttrRs::TexCoord2(attribute.data.map_py(py)?)),
        AttributeType::TexCoord3 => Ok(AttrRs::TexCoord3(attribute.data.map_py(py)?)),
        AttributeType::TexCoord4 => Ok(AttrRs::TexCoord4(attribute.data.map_py(py)?)),
        AttributeType::TexCoord5 => Ok(AttrRs::TexCoord5(attribute.data.map_py(py)?)),
        AttributeType::TexCoord6 => Ok(AttrRs::TexCoord6(attribute.data.map_py(py)?)),
        AttributeType::TexCoord7 => Ok(AttrRs::TexCoord7(attribute.data.map_py(py)?)),
        AttributeType::TexCoord8 => Ok(AttrRs::TexCoord8(attribute.data.map_py(py)?)),
        AttributeType::VertexColor => Ok(AttrRs::VertexColor(attribute.data.map_py(py)?)),
        AttributeType::Blend => Ok(AttrRs::Blend(attribute.data.map_py(py)?)),
        AttributeType::WeightIndex => Ok(AttrRs::WeightIndex(attribute.data.extract(py)?)),
        AttributeType::Position2 => Ok(AttrRs::Position2(attribute.data.map_py(py)?)),
        AttributeType::Normal4 => Ok(AttrRs::Normal4(attribute.data.map_py(py)?)),
        AttributeType::OldPosition => Ok(AttrRs::OldPosition(attribute.data.map_py(py)?)),
        AttributeType::Tangent2 => Ok(AttrRs::Tangent2(attribute.data.map_py(py)?)),
        AttributeType::SkinWeights => Ok(AttrRs::SkinWeights(attribute.data.map_py(py)?)),
        AttributeType::SkinWeights2 => Ok(AttrRs::SkinWeights2(attribute.data.map_py(py)?)),
        AttributeType::BoneIndices => Ok(AttrRs::BoneIndices(attribute.data.extract(py)?)),
        AttributeType::BoneIndices2 => Ok(AttrRs::BoneIndices2(attribute.data.extract(py)?)),
    }
}

fn morph_targets_py(py: Python, targets: Vec<xc3_model::vertex::MorphTarget>) -> Py<PyList> {
    // TODO: avoid unwrap.
    PyList::new_bound(
        py,
        targets.into_iter().map(|target| {
            MorphTarget {
                morph_controller_index: target.morph_controller_index,
                position_deltas: target.position_deltas.map_py(py).unwrap(),
                normals: target.normals.map_py(py).unwrap(),
                tangents: target.tangents.map_py(py).unwrap(),
                vertex_indices: target.vertex_indices.into_pyarray_bound(py).into(),
            }
            .into_py(py)
        }),
    )
    .into()
}

fn index_buffers_py(py: Python, index_buffers: Vec<xc3_model::vertex::IndexBuffer>) -> Py<PyList> {
    PyList::new_bound(
        py,
        index_buffers.into_iter().map(|buffer| {
            IndexBuffer {
                indices: buffer.indices.into_pyarray_bound(py).into(),
            }
            .into_py(py)
        }),
    )
    .into()
}

fn vertex_buffers_py(
    py: Python,
    vertex_buffers: Vec<xc3_model::vertex::VertexBuffer>,
) -> PyResult<Py<PyList>> {
    Ok(PyList::new_bound(
        py,
        vertex_buffers
            .into_iter()
            .map(|buffer| {
                Ok(VertexBuffer {
                    attributes: vertex_attributes_py(py, buffer.attributes)?,
                    morph_blend_target: vertex_attributes_py(py, buffer.morph_blend_target)?,
                    morph_targets: morph_targets_py(py, buffer.morph_targets),
                    outline_buffer_index: buffer.outline_buffer_index,
                }
                .into_py(py))
            })
            .collect::<PyResult<Vec<_>>>()?,
    )
    .into())
}

fn vertex_buffer_rs(py: Python, b: &VertexBuffer) -> PyResult<xc3_model::vertex::VertexBuffer> {
    Ok(xc3_model::vertex::VertexBuffer {
        attributes: b
            .attributes
            .extract::<'_, '_, Vec<AttributeData>>(py)?
            .iter()
            .map(|a| attribute_data_rs(py, a))
            .collect::<PyResult<Vec<_>>>()?,
        morph_blend_target: b
            .morph_blend_target
            .extract::<'_, '_, Vec<AttributeData>>(py)?
            .iter()
            .map(|a| attribute_data_rs(py, a))
            .collect::<PyResult<Vec<_>>>()?,
        morph_targets: b
            .morph_targets
            .extract::<'_, '_, Vec<MorphTarget>>(py)?
            .iter()
            .map(|t| {
                Ok(xc3_model::vertex::MorphTarget {
                    morph_controller_index: t.morph_controller_index,
                    position_deltas: t.position_deltas.map_py(py)?,
                    normals: t.normals.map_py(py)?,
                    tangents: t.tangents.map_py(py)?,
                    vertex_indices: t.vertex_indices.extract(py)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
        outline_buffer_index: b.outline_buffer_index,
    })
}
