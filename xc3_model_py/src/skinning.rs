use pyo3::{prelude::*, types::PyList};

use crate::{map_py::MapPy, uvec2s_pyarray, RenderPassType};

#[pyclass]
#[derive(Debug, Clone)]
pub struct Weights {
    #[pyo3(get, set)]
    weight_buffers: Py<PyList>,
    // TODO: how to handle this?
    weight_groups: xc3_model::skinning::WeightGroups,
}

#[pymethods]
impl Weights {
    #[new]
    pub fn new(weight_buffers: Py<PyList>) -> Self {
        // TODO: rework xc3_model to make this easier to work with.
        Self {
            weight_buffers,
            weight_groups: xc3_model::skinning::WeightGroups::Groups {
                weight_groups: Vec::new(),
                weight_lods: Vec::new(),
            },
        }
    }

    pub fn weight_buffer(&self, py: Python, flags2: u32) -> PyResult<Option<SkinWeights>> {
        self.map_py(py)?
            .weight_buffer(flags2)
            .map(|b| b.map_py(py))
            .transpose()
    }

    // TODO: make this a method of WeightGroups?
    pub fn weights_start_index(
        &self,
        skin_flags: u32,
        lod_item_index: usize,
        unk_type: RenderPassType,
    ) -> usize {
        // TODO: Move the optional parameter last in Rust to match Python?
        self.weight_groups
            .weights_start_index(skin_flags, Some(lod_item_index), unk_type.into())
    }

    pub fn update_weights(&mut self, py: Python, combined_weights: &SkinWeights) -> PyResult<()> {
        let mut weights = self.map_py(py)?;

        let combined_weights = combined_weights.map_py(py)?;
        weights.update_weights(combined_weights);

        *self = weights.map_py(py)?;
        Ok(())
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::skinning::SkinWeights)]
pub struct SkinWeights {
    // N x 4 numpy.ndarray
    pub bone_indices: PyObject,
    // N x 4 numpy.ndarray
    pub weights: PyObject,
    /// The name list for the indices in [bone_indices](#structfield.bone_indices).
    pub bone_names: Py<PyList>,
}

#[pymethods]
impl SkinWeights {
    #[new]
    pub fn new(bone_indices: PyObject, weights: PyObject, bone_names: Py<PyList>) -> Self {
        Self {
            bone_indices,
            weights,
            bone_names,
        }
    }

    pub fn to_influences(&self, py: Python, weight_indices: PyObject) -> PyResult<Py<PyList>> {
        let weight_indices: Vec<_> = weight_indices.extract(py)?;
        let influences = self.map_py(py)?.to_influences(&weight_indices);
        influences.map_py(py)
    }

    pub fn add_influences(
        &mut self,
        py: Python,
        influences: Vec<PyRef<Influence>>,
        vertex_count: usize,
    ) -> PyResult<PyObject> {
        let influences = influences
            .iter()
            .map(|i| i.map_py(py))
            .collect::<PyResult<Vec<_>>>()?;
        let mut skin_weights = self.map_py(py)?;
        let weight_indices = skin_weights.add_influences(&influences, vertex_count);
        *self = skin_weights.map_py(py)?;
        Ok(uvec2s_pyarray(py, &weight_indices))
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::skinning::Influence)]
pub struct Influence {
    pub bone_name: String,
    pub weights: Py<PyList>,
}

#[pymethods]
impl Influence {
    #[new]
    fn new(bone_name: String, weights: Py<PyList>) -> Self {
        Self { bone_name, weights }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, MapPy)]
#[map(xc3_model::skinning::VertexWeight)]
pub struct VertexWeight {
    pub vertex_index: u32,
    pub weight: f32,
}

#[pymethods]
impl VertexWeight {
    #[new]
    fn new(vertex_index: u32, weight: f32) -> Self {
        Self {
            vertex_index,
            weight,
        }
    }
}

pub fn skinning(py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "skinning")?;

    m.add_class::<Weights>()?;
    m.add_class::<SkinWeights>()?;
    m.add_class::<Influence>()?;
    m.add_class::<VertexWeight>()?;

    module.add_submodule(&m)?;
    Ok(())
}

impl MapPy<Weights> for xc3_model::skinning::Weights {
    fn map_py(&self, py: Python) -> PyResult<Weights> {
        Ok(Weights {
            weight_buffers: self.weight_buffers.map_py(py)?,
            weight_groups: self.weight_groups.clone(),
        })
    }
}

impl MapPy<xc3_model::skinning::Weights> for Weights {
    fn map_py(&self, py: Python) -> PyResult<xc3_model::skinning::Weights> {
        Ok(xc3_model::skinning::Weights {
            weight_buffers: self.weight_buffers.map_py(py)?,
            weight_groups: self.weight_groups.clone(),
        })
    }
}
