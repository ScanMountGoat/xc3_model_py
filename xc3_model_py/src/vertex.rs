use pyo3::{prelude::*, types::PyList};

use crate::python_enum;

python_enum!(
    PrimitiveType,
    xc3_model::vertex::PrimitiveType,
    TriangleList,
    QuadList,
    TriangleStrip,
    TriangleListAdjacency
);

#[pymodule]
pub mod vertex {
    use super::*;

    use crate::{map_py::MapPy, skinning::skinning::Weights};
    use numpy::{PyArray1, PyArray2, PyUntypedArray};

    #[pymodule_export]
    use super::PrimitiveType;

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::vertex::ModelBuffers)]
    pub struct ModelBuffers {
        pub vertex_buffers: Py<PyList>,
        pub outline_buffers: Py<PyList>,
        pub index_buffers: Py<PyList>,
        pub unk_buffers: Py<PyList>,
        pub unk_data: Option<Py<UnkDataBuffer>>,
        pub weights: Option<Py<Weights>>,
    }

    #[pymethods]
    impl ModelBuffers {
        #[new]
        pub fn new(
            vertex_buffers: Py<PyList>,
            outline_buffers: Py<PyList>,
            index_buffers: Py<PyList>,
            unk_buffers: Py<PyList>,
            unk_data: Option<Py<UnkDataBuffer>>,
            weights: Option<Py<Weights>>,
        ) -> Self {
            Self {
                vertex_buffers,
                outline_buffers,
                index_buffers,
                unk_buffers,
                unk_data,
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
        pub indices: Py<PyArray1<u16>>,
        pub primitive_type: PrimitiveType,
    }

    #[pymethods]
    impl IndexBuffer {
        #[new]
        fn new(indices: Py<PyArray1<u16>>, primitive_type: PrimitiveType) -> Self {
            Self {
                indices,
                primitive_type,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::vertex::OutlineBuffer)]
    pub struct OutlineBuffer {
        pub attributes: Py<PyList>,
    }

    #[pymethods]
    impl OutlineBuffer {
        #[new]
        fn new(attributes: Py<PyList>) -> Self {
            Self { attributes }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct AttributeData {
        pub attribute_type: AttributeType,
        // numpy.ndarray with vertex count many rows
        pub data: Py<PyUntypedArray>,
    }

    #[pymethods]
    impl AttributeData {
        #[new]
        fn new(attribute_type: AttributeType, data: Py<PyUntypedArray>) -> Self {
            Self {
                attribute_type,
                data,
            }
        }
    }

    #[pyclass(eq, eq_int)]
    #[derive(Debug, PartialEq, Eq, Clone)]
    pub enum AttributeType {
        Position,
        SkinWeights2,
        BoneIndices2,
        WeightIndex,
        WeightIndex2,
        TexCoord0,
        TexCoord1,
        TexCoord2,
        TexCoord3,
        TexCoord4,
        TexCoord5,
        TexCoord6,
        TexCoord7,
        TexCoord8,
        Blend,
        Unk15,
        Unk16,
        VertexColor,
        Unk18,
        Unk24,
        Unk25,
        Unk26,
        Normal,
        Tangent,
        Unk30,
        Unk31,
        Normal2,
        ValInf,
        Normal3,
        VertexColor3,
        Position2,
        Normal4,
        OldPosition,
        Tangent2,
        SkinWeights,
        BoneIndices,
        Flow,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::vertex::MorphTarget)]
    pub struct MorphTarget {
        pub morph_controller_index: usize,
        // N x 3 numpy.ndarray
        pub position_deltas: Py<PyArray2<f32>>,
        // N x 4 numpy.ndarray
        pub normals: Py<PyArray2<f32>>,
        // N x 4 numpy.ndarray
        pub tangents: Py<PyArray2<f32>>,
        pub vertex_indices: Py<PyArray1<u32>>,
    }

    #[pymethods]
    impl MorphTarget {
        #[new]
        fn new(
            morph_controller_index: usize,
            position_deltas: Py<PyArray2<f32>>,
            normals: Py<PyArray2<f32>>,
            tangents: Py<PyArray2<f32>>,
            vertex_indices: Py<PyArray1<u32>>,
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

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::vertex::UnkBuffer)]
    pub struct UnkBuffer {
        pub unk2: u16,
        pub attributes: Py<PyList>,
    }

    #[pymethods]
    impl UnkBuffer {
        #[new]
        fn new(unk2: u16, attributes: Py<PyList>) -> Self {
            Self { unk2, attributes }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::vertex::UnkDataBuffer)]
    pub struct UnkDataBuffer {
        pub attribute1: Py<PyArray2<u32>>,
        pub attribute2: Py<PyArray1<u32>>,
        pub uniform_data: Vec<u8>,
        pub unk: [f32; 6],
    }

    #[pymethods]
    impl UnkDataBuffer {
        #[new]
        fn new(
            attribute1: Py<PyArray2<u32>>,
            attribute2: Py<PyArray1<u32>>,
            uniform_data: Vec<u8>,
            unk: [f32; 6],
        ) -> Self {
            Self {
                attribute1,
                attribute2,
                uniform_data,
                unk,
            }
        }
    }

    impl MapPy<AttributeData> for xc3_model::vertex::AttributeData {
        fn map_py(self, py: Python) -> PyResult<AttributeData> {
            match self {
                xc3_model::vertex::AttributeData::Position(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Position,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::SkinWeights2(values) => Ok(AttributeData {
                    attribute_type: AttributeType::SkinWeights2,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::BoneIndices2(values) => Ok(AttributeData {
                    attribute_type: AttributeType::BoneIndices2,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::WeightIndex(values) => Ok(AttributeData {
                    attribute_type: AttributeType::WeightIndex,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::WeightIndex2(values) => Ok(AttributeData {
                    attribute_type: AttributeType::WeightIndex2,
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
                xc3_model::vertex::AttributeData::Blend(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Blend,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Unk15(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Unk15,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Unk16(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Unk16,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::VertexColor(values) => Ok(AttributeData {
                    attribute_type: AttributeType::VertexColor,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Unk18(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Unk18,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Unk24(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Unk24,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Unk25(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Unk25,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Unk26(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Unk26,
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
                xc3_model::vertex::AttributeData::Unk30(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Unk30,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Unk31(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Unk31,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Normal2(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Normal2,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::ValInf(values) => Ok(AttributeData {
                    attribute_type: AttributeType::ValInf,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Normal3(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Normal3,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::VertexColor3(values) => Ok(AttributeData {
                    attribute_type: AttributeType::VertexColor3,
                    data: values.map_py(py)?,
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
                xc3_model::vertex::AttributeData::BoneIndices(values) => Ok(AttributeData {
                    attribute_type: AttributeType::BoneIndices,
                    data: values.map_py(py)?,
                }),
                xc3_model::vertex::AttributeData::Flow(values) => Ok(AttributeData {
                    attribute_type: AttributeType::Flow,
                    data: values.map_py(py)?,
                }),
            }
        }
    }

    impl MapPy<xc3_model::vertex::AttributeData> for AttributeData {
        fn map_py(self, py: Python) -> PyResult<xc3_model::vertex::AttributeData> {
            use xc3_model::vertex::AttributeData as AttrRs;
            match self.attribute_type {
                AttributeType::Position => Ok(AttrRs::Position(self.data.map_py(py)?)),
                AttributeType::SkinWeights2 => Ok(AttrRs::SkinWeights2(self.data.map_py(py)?)),
                AttributeType::BoneIndices2 => Ok(AttrRs::BoneIndices2(self.data.map_py(py)?)),
                AttributeType::WeightIndex => Ok(AttrRs::WeightIndex(self.data.map_py(py)?)),
                AttributeType::WeightIndex2 => Ok(AttrRs::WeightIndex2(self.data.map_py(py)?)),
                AttributeType::TexCoord0 => Ok(AttrRs::TexCoord0(self.data.map_py(py)?)),
                AttributeType::TexCoord1 => Ok(AttrRs::TexCoord1(self.data.map_py(py)?)),
                AttributeType::TexCoord2 => Ok(AttrRs::TexCoord2(self.data.map_py(py)?)),
                AttributeType::TexCoord3 => Ok(AttrRs::TexCoord3(self.data.map_py(py)?)),
                AttributeType::TexCoord4 => Ok(AttrRs::TexCoord4(self.data.map_py(py)?)),
                AttributeType::TexCoord5 => Ok(AttrRs::TexCoord5(self.data.map_py(py)?)),
                AttributeType::TexCoord6 => Ok(AttrRs::TexCoord6(self.data.map_py(py)?)),
                AttributeType::TexCoord7 => Ok(AttrRs::TexCoord7(self.data.map_py(py)?)),
                AttributeType::TexCoord8 => Ok(AttrRs::TexCoord8(self.data.map_py(py)?)),
                AttributeType::Blend => Ok(AttrRs::Blend(self.data.map_py(py)?)),
                AttributeType::Unk15 => Ok(AttrRs::Unk15(self.data.map_py(py)?)),
                AttributeType::Unk16 => Ok(AttrRs::Unk16(self.data.map_py(py)?)),
                AttributeType::VertexColor => Ok(AttrRs::VertexColor(self.data.map_py(py)?)),
                AttributeType::Unk18 => Ok(AttrRs::Unk18(self.data.map_py(py)?)),
                AttributeType::Unk24 => Ok(AttrRs::Unk24(self.data.map_py(py)?)),
                AttributeType::Unk25 => Ok(AttrRs::Unk25(self.data.map_py(py)?)),
                AttributeType::Unk26 => Ok(AttrRs::Unk26(self.data.map_py(py)?)),
                AttributeType::Normal => Ok(AttrRs::Normal(self.data.map_py(py)?)),
                AttributeType::Tangent => Ok(AttrRs::Tangent(self.data.map_py(py)?)),
                AttributeType::Unk30 => Ok(AttrRs::Unk30(self.data.map_py(py)?)),
                AttributeType::Unk31 => Ok(AttrRs::Unk31(self.data.map_py(py)?)),
                AttributeType::Normal2 => Ok(AttrRs::Normal2(self.data.map_py(py)?)),
                AttributeType::ValInf => Ok(AttrRs::ValInf(self.data.map_py(py)?)),
                AttributeType::Normal3 => Ok(AttrRs::Normal3(self.data.map_py(py)?)),
                AttributeType::VertexColor3 => Ok(AttrRs::VertexColor3(self.data.map_py(py)?)),
                AttributeType::Position2 => Ok(AttrRs::Position2(self.data.map_py(py)?)),
                AttributeType::Normal4 => Ok(AttrRs::Normal4(self.data.map_py(py)?)),
                AttributeType::OldPosition => Ok(AttrRs::OldPosition(self.data.map_py(py)?)),
                AttributeType::Tangent2 => Ok(AttrRs::Tangent2(self.data.map_py(py)?)),
                AttributeType::SkinWeights => Ok(AttrRs::SkinWeights(self.data.map_py(py)?)),
                AttributeType::BoneIndices => Ok(AttrRs::BoneIndices(self.data.map_py(py)?)),
                AttributeType::Flow => Ok(AttrRs::Flow(self.data.map_py(py)?)),
            }
        }
    }

    // Map from Python lists to Vec<T>
    impl crate::MapPy<Vec<xc3_model::vertex::AttributeData>> for Py<PyList> {
        fn map_py(self, py: Python) -> PyResult<Vec<xc3_model::vertex::AttributeData>> {
            self.extract::<'_, '_, Vec<AttributeData>>(py)?
                .into_iter()
                .map(|v| v.map_py(py))
                .collect::<Result<Vec<_>, _>>()
        }
    }

    // Map from Vec<T> to Python lists
    impl crate::MapPy<Py<PyList>> for Vec<xc3_model::vertex::AttributeData> {
        fn map_py(self, py: Python) -> PyResult<Py<PyList>> {
            PyList::new(
                py,
                self.into_iter()
                    .map(|v| v.map_py(py)?.into_pyobject(py))
                    .collect::<PyResult<Vec<_>>>()?,
            )
            .map(Into::into)
        }
    }
}
