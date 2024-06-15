use pyo3::prelude::*;
use xc3_model::animation::BoneIndex;

use crate::{map_py::MapPy, mat4_to_pyarray, python_enum, transforms_pyarray, Skeleton};

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct Animation {
    pub name: String,
    pub space_mode: SpaceMode,
    pub play_mode: PlayMode,
    pub blend_mode: BlendMode,
    pub frames_per_second: f32,
    pub frame_count: u32,
    pub tracks: Vec<Track>,
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
    ) -> PyResult<PyObject> {
        let animation = animation_rs(self);
        let skeleton = skeleton.map_py(py)?;
        let transforms = animation.skinning_transforms(&skeleton, frame);
        Ok(transforms_pyarray(py, &transforms))
    }

    pub fn model_space_transforms(
        &self,
        py: Python,
        skeleton: Skeleton,
        frame: f32,
    ) -> PyResult<PyObject> {
        let animation = animation_rs(self);
        let skeleton = skeleton.map_py(py)?;
        let transforms = animation.model_space_transforms(&skeleton, frame);
        Ok(transforms_pyarray(py, &transforms))
    }

    pub fn local_space_transforms(
        &self,
        py: Python,
        skeleton: Skeleton,
        frame: f32,
    ) -> PyResult<PyObject> {
        let animation = animation_rs(self);
        let skeleton = skeleton.map_py(py)?;
        let transforms = animation.local_space_transforms(&skeleton, frame);
        Ok(transforms_pyarray(py, &transforms))
    }
}

// TODO: Expose implementation details?
#[pyclass]
#[derive(Debug, Clone)]
pub struct Track(xc3_model::animation::Track);

#[pymethods]
impl Track {
    pub fn sample_translation(&self, frame: f32, frame_count: u32) -> Option<(f32, f32, f32)> {
        self.0
            .sample_translation(frame, frame_count)
            .map(Into::into)
    }

    pub fn sample_rotation(&self, frame: f32, frame_count: u32) -> Option<(f32, f32, f32, f32)> {
        self.0.sample_rotation(frame, frame_count).map(Into::into)
    }

    pub fn sample_scale(&self, frame: f32, frame_count: u32) -> Option<(f32, f32, f32)> {
        self.0.sample_scale(frame, frame_count).map(Into::into)
    }

    pub fn sample_transform(&self, py: Python, frame: f32, frame_count: u32) -> Option<PyObject> {
        self.0
            .sample_transform(frame, frame_count)
            .map(|t| mat4_to_pyarray(py, t))
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
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub x_coeffs: (f32, f32, f32, f32),
    pub y_coeffs: (f32, f32, f32, f32),
    pub z_coeffs: (f32, f32, f32, f32),
    pub w_coeffs: (f32, f32, f32, f32),
}

python_enum!(SpaceMode, xc3_model::animation::SpaceMode, Local, Model);
python_enum!(PlayMode, xc3_model::animation::PlayMode, Loop, Single);
python_enum!(BlendMode, xc3_model::animation::BlendMode, Blend, Add);

#[pyfunction]
fn murmur3(name: &str) -> u32 {
    xc3_model::animation::murmur3(name.as_bytes())
}

pub fn animation(py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "animation")?;

    m.add_class::<Animation>()?;
    m.add_class::<Track>()?;
    m.add_class::<Keyframe>()?;
    m.add_class::<SpaceMode>()?;
    m.add_class::<PlayMode>()?;
    m.add_class::<BlendMode>()?;
    m.add_function(wrap_pyfunction!(murmur3, &m)?)?;

    module.add_submodule(&m)?;
    Ok(())
}

pub fn animation_rs(animation: &Animation) -> xc3_model::animation::Animation {
    xc3_model::animation::Animation {
        name: animation.name.clone(),
        space_mode: animation.space_mode.into(),
        play_mode: animation.play_mode.into(),
        blend_mode: animation.blend_mode.into(),
        frames_per_second: animation.frames_per_second,
        frame_count: animation.frame_count,
        tracks: animation.tracks.iter().map(|t| t.0.clone()).collect(),
        morph_tracks: None, // TODO: morph animations?
    }
}

pub fn animation_py(animation: xc3_model::animation::Animation) -> Animation {
    Animation {
        name: animation.name.clone(),
        space_mode: animation.space_mode.into(),
        play_mode: animation.play_mode.into(),
        blend_mode: animation.blend_mode.into(),
        frames_per_second: animation.frames_per_second,
        frame_count: animation.frame_count,
        tracks: animation.tracks.into_iter().map(Track).collect(),
    }
}
