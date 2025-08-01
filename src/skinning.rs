use pyo3::prelude::*;

use crate::python_enum;

python_enum!(
    BoneConstraintType,
    xc3_model::skinning::BoneConstraintType,
    FixedOffset,
    Distance
);

#[pymodule]
pub mod skinning {
    use numpy::PyArray2;
    use pyo3::prelude::*;

    use crate::material::RenderPassType;
    use map_py::{MapPy, TypedList};

    #[pymodule_export]
    use super::BoneConstraintType;

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::skinning::Skinning)]
    pub struct Skinning {
        bones: TypedList<Bone>,
    }

    #[pymethods]
    impl Skinning {
        #[new]
        pub fn new(bones: TypedList<Bone>) -> Self {
            Self { bones }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::skinning::Bone)]
    pub struct Bone {
        name: String,

        #[map(from(map_py::helpers::into_option_py))]
        #[map(into(map_py::helpers::from_option_py))]
        bounds: Option<Py<BoneBounds>>,

        #[map(from(map_py::helpers::into_option_py))]
        #[map(into(map_py::helpers::from_option_py))]
        constraint: Option<Py<BoneConstraint>>,

        no_camera_overlap: bool,
    }

    #[pymethods]
    impl Bone {
        #[new]
        pub fn new(
            name: String,
            no_camera_overlap: bool,
            bounds: Option<Py<BoneBounds>>,
            constraint: Option<Py<BoneConstraint>>,
        ) -> Self {
            Self {
                name,
                bounds,
                constraint,
                no_camera_overlap,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::skinning::BoneBounds)]
    pub struct BoneBounds {
        center: [f32; 3],
        size: [f32; 3],
        radius: f32,
    }

    #[pymethods]
    impl BoneBounds {
        #[new]
        pub fn new(center: [f32; 3], size: [f32; 3], radius: f32) -> Self {
            Self {
                center,
                size,
                radius,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::skinning::BoneConstraint)]
    pub struct BoneConstraint {
        fixed_offset: [f32; 3],
        max_distance: f32,
        constraint_type: BoneConstraintType,
        parent_index: Option<usize>,
    }

    #[pymethods]
    impl BoneConstraint {
        #[new]
        pub fn new(
            fixed_offset: [f32; 3],
            max_distance: f32,
            constraint_type: BoneConstraintType,
            parent_index: Option<usize>,
        ) -> Self {
            Self {
                fixed_offset,
                max_distance,
                constraint_type,
                parent_index,
            }
        }
    }

    #[pyclass]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::skinning::Weights)]
    pub struct Weights {
        #[pyo3(get, set)]
        #[map(from(MapPy::map_py), into(MapPy::map_py))]
        weight_buffers: TypedList<SkinWeights>,
        // TODO: how to handle this?
        #[map(from(map_py::helpers::into), into(map_py::helpers::into))]
        weight_groups: xc3_model::skinning::WeightGroups,
    }

    #[pymethods]
    impl Weights {
        #[new]
        pub fn new(weight_buffers: TypedList<SkinWeights>) -> Self {
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
            self.clone()
                .map_py(py)?
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
            self.weight_groups.weights_start_index(
                skin_flags,
                Some(lod_item_index),
                unk_type.into(),
            )
        }

        pub fn update_weights(
            &mut self,
            py: Python,
            combined_weights: &SkinWeights,
        ) -> PyResult<()> {
            let mut weights = self.clone().map_py(py)?;

            let combined_weights = combined_weights.clone().map_py(py)?;
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
        pub bone_indices: Py<PyArray2<u8>>,
        // N x 4 numpy.ndarray
        pub weights: Py<PyArray2<f32>>,
        /// The name list for the indices in [bone_indices](#structfield.bone_indices).
        pub bone_names: TypedList<String>,
    }

    #[pymethods]
    impl SkinWeights {
        #[new]
        pub fn new(
            bone_indices: Py<PyArray2<u8>>,
            weights: Py<PyArray2<f32>>,
            bone_names: TypedList<String>,
        ) -> Self {
            Self {
                bone_indices,
                weights,
                bone_names,
            }
        }

        pub fn to_influences(
            &self,
            py: Python,
            weight_indices: Py<PyArray2<u16>>,
        ) -> PyResult<TypedList<Influence>> {
            let weight_indices: Vec<_> = weight_indices.extract(py)?;
            let influences = self.clone().map_py(py)?.to_influences(&weight_indices);
            influences.map_py(py)
        }

        pub fn add_influences(
            &mut self,
            py: Python,
            influences: Vec<PyRef<Influence>>,
            vertex_count: usize,
        ) -> PyResult<Py<PyArray2<u16>>> {
            let influences = influences
                .into_iter()
                .map(|i| {
                    let i: &Influence = &i;
                    i.clone().map_py(py)
                })
                .collect::<PyResult<Vec<_>>>()?;
            let mut skin_weights = self.clone().map_py(py)?;
            let weight_indices = skin_weights.add_influences(&influences, vertex_count);
            *self = skin_weights.map_py(py)?;
            weight_indices.map_py(py)
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::skinning::Influence)]
    pub struct Influence {
        pub bone_name: String,
        pub weights: TypedList<VertexWeight>,
    }

    #[pymethods]
    impl Influence {
        #[new]
        fn new(bone_name: String, weights: TypedList<VertexWeight>) -> Self {
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
}
