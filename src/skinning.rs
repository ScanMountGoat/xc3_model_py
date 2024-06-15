use pyo3::{prelude::*, types::PyList};

use crate::{pyarray_to_vec4s, uvec2s_pyarray, uvec4_pyarray, vec4s_pyarray, RenderPassType};

#[pyclass]
#[derive(Debug, Clone)]
pub struct Weights {
    #[pyo3(get, set)]
    weight_buffers: Vec<SkinWeights>,
    // TODO: how to handle this?
    weight_groups: xc3_model::skinning::WeightGroups,
}

#[pymethods]
impl Weights {
    #[new]
    pub fn new(weight_buffers: Vec<SkinWeights>) -> Self {
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
        Ok(weights_rs(py, self)?
            .weight_buffer(flags2)
            .map(|b| skin_weights_py(py, b)))
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
        let mut weights = weights_rs(py, self)?;

        let combined_weights = skin_weights_rs(py, combined_weights)?;
        weights.update_weights(combined_weights);

        *self = weights_py(py, weights);
        Ok(())
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
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

    pub fn to_influences(&self, py: Python, weight_indices: PyObject) -> PyResult<Vec<Influence>> {
        let weight_indices: Vec<_> = weight_indices.extract(py)?;
        let influences = skin_weights_rs(py, self)?.to_influences(&weight_indices);
        Ok(influences_py(py, influences))
    }

    pub fn add_influences(
        &mut self,
        py: Python,
        influences: Vec<PyRef<Influence>>,
        vertex_count: usize,
    ) -> PyResult<PyObject> {
        let influences = influences
            .iter()
            .map(|i| influence_rs(py, i))
            .collect::<PyResult<Vec<_>>>()?;
        let mut skin_weights = skin_weights_rs(py, self)?;
        let weight_indices = skin_weights.add_influences(&influences, vertex_count);
        *self = skin_weights_py(py, skin_weights);
        Ok(uvec2s_pyarray(py, &weight_indices))
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

fn weight_buffers_rs(
    py: Python,
    weight_buffers: &[SkinWeights],
) -> Result<Vec<xc3_model::skinning::SkinWeights>, PyErr> {
    weight_buffers
        .iter()
        .map(|w| skin_weights_rs(py, w))
        .collect()
}

fn skin_weights_rs(py: Python, w: &SkinWeights) -> PyResult<xc3_model::skinning::SkinWeights> {
    Ok(xc3_model::skinning::SkinWeights {
        bone_indices: w.bone_indices.extract(py)?,
        weights: pyarray_to_vec4s(py, &w.weights)?,
        bone_names: w.bone_names.extract(py)?,
    })
}

fn skin_weights_py(py: Python, w: xc3_model::skinning::SkinWeights) -> SkinWeights {
    SkinWeights {
        bone_indices: uvec4_pyarray(py, &w.bone_indices),
        weights: vec4s_pyarray(py, &w.weights),
        bone_names: PyList::new_bound(py, &w.bone_names).into(),
    }
}

fn weight_buffers_py(
    py: Python,
    weight_buffers: Vec<xc3_model::skinning::SkinWeights>,
) -> Vec<SkinWeights> {
    weight_buffers
        .into_iter()
        .map(|w| skin_weights_py(py, w))
        .collect()
}

pub fn weights_py(py: Python, weights: xc3_model::skinning::Weights) -> Weights {
    Weights {
        weight_buffers: weight_buffers_py(py, weights.weight_buffers),
        weight_groups: weights.weight_groups,
    }
}

pub fn weights_rs(py: Python, w: &Weights) -> PyResult<xc3_model::skinning::Weights> {
    Ok(xc3_model::skinning::Weights {
        weight_buffers: weight_buffers_rs(py, &w.weight_buffers)?,
        weight_groups: w.weight_groups.clone(),
    })
}

fn influences_py(py: Python, influences: Vec<xc3_model::skinning::Influence>) -> Vec<Influence> {
    influences
        .into_iter()
        .map(|i| Influence {
            bone_name: i.bone_name,
            weights: PyList::new_bound(
                py,
                i.weights.into_iter().map(|w| {
                    VertexWeight {
                        vertex_index: w.vertex_index,
                        weight: w.weight,
                    }
                    .into_py(py)
                }),
            )
            .into(),
        })
        .collect()
}

fn influence_rs(py: Python, influence: &Influence) -> PyResult<xc3_model::skinning::Influence> {
    Ok(xc3_model::skinning::Influence {
        bone_name: influence.bone_name.clone(),
        weights: influence
            .weights
            .extract::<Vec<VertexWeight>>(py)?
            .into_iter()
            .map(|w| xc3_model::skinning::VertexWeight {
                vertex_index: w.vertex_index,
                weight: w.weight,
            })
            .collect(),
    })
}
