use pyo3::prelude::*;

use crate::python_enum;

python_enum!(SpaceMode, xc3_model::animation::SpaceMode, Local, Model);
python_enum!(PlayMode, xc3_model::animation::PlayMode, Loop, Single);
python_enum!(BlendMode, xc3_model::animation::BlendMode, Blend, Add);

#[pymodule]
pub mod animation {
    use crate::xc3_model_py::Skeleton;
    use map_py::MapPy;
    use numpy::PyArray1;
    use numpy::PyArray2;
    use numpy::PyArray3;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use xc3_model::animation::BoneIndex;

    #[pymodule_export]
    use super::SpaceMode;

    #[pymodule_export]
    use super::PlayMode;

    #[pymodule_export]
    use super::BlendMode;

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::animation::Animation)]
    pub struct Animation {
        pub name: String,
        pub space_mode: SpaceMode,
        pub play_mode: PlayMode,
        pub blend_mode: BlendMode,
        pub frames_per_second: f32,
        pub frame_count: u32,
        pub tracks: Vec<Track>,
        #[map(from(map_py::helpers::into_option_py))]
        #[map(into(map_py::helpers::from_option_py))]
        pub morph_tracks: Option<Py<MorphTracks>>,
        pub root_translation: Option<Py<PyArray2<f32>>>,
    }

    #[pymethods]
    impl Animation {
        pub fn current_frame(&self, current_time_seconds: f32) -> f32 {
            // TODO: looping?
            current_time_seconds * self.frames_per_second
        }

        pub fn skinning_transforms(
            &self,
            py: Python,
            skeleton: Skeleton,
            frame: f32,
        ) -> PyResult<Py<PyArray3<f32>>> {
            let animation: xc3_model::animation::Animation = self.clone().map_py(py)?;
            let skeleton = skeleton.map_py(py)?;
            let transforms = animation.skinning_transforms(&skeleton, frame);
            transforms.map_py(py)
        }

        pub fn model_space_transforms(
            &self,
            py: Python,
            skeleton: Skeleton,
            frame: f32,
        ) -> PyResult<Py<PyArray3<f32>>> {
            let animation: xc3_model::animation::Animation = self.clone().map_py(py)?;
            let skeleton = skeleton.map_py(py)?;
            let transforms = animation.model_space_transforms(&skeleton, frame);
            let matrices: Vec<_> = transforms.into_iter().map(|t| t.to_matrix()).collect();
            matrices.map_py(py)
        }

        pub fn local_space_transforms(
            &self,
            py: Python,
            skeleton: Skeleton,
            frame: f32,
        ) -> PyResult<Py<PyArray3<f32>>> {
            let animation: xc3_model::animation::Animation = self.clone().map_py(py)?;
            let skeleton = skeleton.map_py(py)?;
            let transforms = animation.local_space_transforms(&skeleton, frame);
            transforms.map_py(py)
        }

        pub fn fcurves(
            &self,
            py: Python,
            skeleton: Skeleton,
            use_blender_coordinates: bool,
        ) -> PyResult<FCurves> {
            let animation: xc3_model::animation::Animation = self.clone().map_py(py)?;
            let skeleton = skeleton.map_py(py)?;
            let fcurves = animation.fcurves(&skeleton, use_blender_coordinates);
            fcurves_py(py, &fcurves)
        }
    }

    // TODO: Expose implementation details?
    #[pyclass]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::animation::Track)]
    pub struct Track(xc3_model::animation::Track);

    #[pymethods]
    impl Track {
        pub fn sample_translation(&self, frame: f32, frame_count: u32) -> Option<(f32, f32, f32)> {
            self.0
                .sample_translation(frame, frame_count)
                .map(Into::into)
        }

        pub fn sample_rotation(
            &self,
            frame: f32,
            frame_count: u32,
        ) -> Option<(f32, f32, f32, f32)> {
            self.0.sample_rotation(frame, frame_count).map(Into::into)
        }

        pub fn sample_scale(&self, frame: f32, frame_count: u32) -> Option<(f32, f32, f32)> {
            self.0.sample_scale(frame, frame_count).map(Into::into)
        }

        pub fn sample_transform(
            &self,
            py: Python,
            frame: f32,
            frame_count: u32,
        ) -> Option<Py<crate::xc3_model_py::Transform>> {
            let transform = self.0.sample_transform(frame, frame_count);
            map_py::helpers::into_option_py(transform, py).unwrap()
        }

        // Workaround for representing Rust enums in Python.
        pub fn bone_index(&self) -> Option<usize> {
            match &self.0.bone_index {
                BoneIndex::Index(index) => Some(*index),
                _ => None,
            }
        }

        pub fn bone_hash(&self) -> Option<u32> {
            match &self.0.bone_index {
                BoneIndex::Hash(hash) => Some(*hash),
                _ => None,
            }
        }

        pub fn bone_name(&self) -> Option<&str> {
            match &self.0.bone_index {
                BoneIndex::Name(name) => Some(name),
                _ => None,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(xc3_model::animation::MorphTracks)]
    pub struct MorphTracks {
        pub track_indices: Py<PyArray1<i16>>,
        pub track_values: Py<PyArray1<f32>>,
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct Keyframe {
        pub x_coeffs: (f32, f32, f32, f32),
        pub y_coeffs: (f32, f32, f32, f32),
        pub z_coeffs: (f32, f32, f32, f32),
        pub w_coeffs: (f32, f32, f32, f32),
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone)]
    pub struct FCurves {
        pub translation: Py<PyDict>,
        pub rotation: Py<PyDict>,
        pub scale: Py<PyDict>,
    }

    #[pyfunction]
    fn murmur3(name: &str) -> u32 {
        xc3_model::animation::murmur3(name.as_bytes())
    }

    pub fn fcurves_py(py: Python, fcurves: &xc3_model::animation::FCurves) -> PyResult<FCurves> {
        let translation = PyDict::new(py);
        for (k, v) in &fcurves.translation {
            let v: Py<PyArray2<f32>> = v.clone().map_py(py)?;
            translation.set_item(k.to_string(), v.into_pyobject(py)?)?;
        }

        let rotation = PyDict::new(py);
        for (k, v) in &fcurves.rotation {
            let v: Py<PyArray2<f32>> = v.clone().map_py(py)?;
            rotation.set_item(k.to_string(), v.into_pyobject(py)?)?;
        }

        let scale = PyDict::new(py);
        for (k, v) in &fcurves.scale {
            let v: Py<PyArray2<f32>> = v.clone().map_py(py)?;
            scale.set_item(k.to_string(), v.into_pyobject(py)?)?;
        }

        Ok(FCurves {
            translation: translation.into(),
            rotation: rotation.into(),
            scale: scale.into(),
        })
    }
}
