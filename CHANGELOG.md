# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## unreleased
### Added
* Added field `ext_mesh_index` to `Mesh`

### Changed
* Adjusted the signature of `animation.Track.sample_` methods to also take the frame count.

### 0.5.1 - 2024-04-29
### Fixed
* Fixed an issue with anim loading for Xenoblade 1 animation files.

### 0.5.0 - 2024-04-29
### Added
* Added function `load_model_legacy` for loading Xenoblade X `.camdo` models.
* Added method `model_space_transforms` to `Skeleton`.
* Added method `model_space_transforms` to `Animation`.
* Added method `local_space_transforms` to `Animation`.
* Added method `skinning_transforms` to `Animation`.
* Added method `sample_transform` to `Track`.
* Added method `to_influences` to `SkinWeights`.
* Added method `to_mxmd_model` to `ModelRoot`.
* Added method `weight_buffer` to `skinning.Weights`.
* Added method `save_images_rgba8` to `ModelRoot` for saving to formats like PNG.
* Added field `texcoord_name` to `ChannelAssignmentTexture`.
* Added field `sampler_index` to `Texture`.
* Added field `mip_filter` to `Sampler`.
* Added fields `max_xyz`, `min_xyz`, `morph_controller_names`, and `animation_morph_names` to `Models`.
* Added fields `max_xyz`, `min_xyz`, and `bounding_radius` to `Model`.
* Added field `flags1` to `Mesh`.
* Added field `vertex_indices` and `morph_controller_index` to `MorphTarget`.
* Added field `outline_buffer_index` to `VertexBuffer`
* Added constructors for all non opaque wrapper types.
* Added opaque wrapper types `Mxmd` and `Msrd` for the types in xc3_lib.
* Added setters for all public fields for all types.

### Changed
* Changed the `sample_translation`, `sample_rotation`, and `sample_scale` for `Track` to return `None` if empty.
* Split `ModelRoot` into separate `ModelRoot` and `MapRoot` types to better reflect in game data.
* Renamed `SkinWeight` to `VertexWeight` to match xc3_model.
* Renamed field `skin_flags` to `flags` for `Mesh`.
* Move animation types and `murmur3` function to `xc3_model_py.animation` submodule.
* Moved `Weights` to `skinning.Weights` and adjusted fields to match xc3_model.
* Changed `AttributeType.WeightIndex` to use a numpy array of u16 with shape (N, 2).
* Changed non bytes or `numpy.ndarray` lists to use pure Python lists to to allow mutating elements.

### Fixed
* Fixed an issue where method `output_assignments` for `Material` did not properly account for textures.

## 0.4.0 - 2024-03-15
### Added
* Added `OutputAssignments`, `OutputAssignment`, `ChannelAssignment`, and `ChanelAssignmentTexture`.
* Added method `output_assignments` to `Material` for accessing shader JSON database information.

### Changed
* Update xc3_model to 0.7.0.
* Changed the types of `MaterialParameters` fields to match xc3_model changes.
* Renamed `Material.unk_type` to `Material.pass_type` to match xc3_model.

### Removed
* Removed `BufferParameter`

## 0.3.0 - 2024-02-28
### Added
* Added support for accessing Rust logs from Python's `logging` module.
* Added `Xc3ModelError` exception type to represent Rust errors.

### Changed
* Update xc3_model to 0.6.0.
* Moved field `skeleton` from `Models` to `ModelRoot`.

### Removed
* Removed methods `sampler_channel_index`, `float_constant`, and `buffer_parameter` from `Shader`.

## 0.2.0 - 2024-01-28
### Added
* Added field `usage` of type `TextureUsage` to `ImageTexture`
* Added `RenderPassType` and `TextureUsage` types
* Added field `skin_flags` of type `int` to `Mesh`
* Added method `weights_start_index` to `Weights`

### Changed
* Updated xc3_model to 0.5.0

## 0.1.0 - 2023-12-06
First release! 