# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.7.0 - 2024-07-22
### Added
* Added type `ChannelAssignmentAttribute` for vertex attribute shader dependencies.
* Added method `attribute` to `ChannelAssignment`.
* Added fields `flags`, `render_flags`, `state_flags`, `work_values`, `shader_vars`, `work_callbacks`, `alpha_test_ref`, `m_unks1_1`, `m_unks1_2`, `m_unks1_3`, `m_unks1_4`, `technique_index`, `parameters`, `m_unks2_2`, and `m_unks3_1` to `Material`.
* Added shader database types in the `shader_database` module for querying the database information.
* Added type `ColorWriteMode` for material state flags.

### Changed
* Renamed `Mesh.unk_mesh_index1` to `Mesh.index_buffer_index2` to better reflect in game data.
* Changed the behavior of `ModelRoot.to_mxmd_model` to also recreate materials.
* Changed the output of exceptions to include inner errors for easier debugging.
* Renamed `ChannelAssignmentTexture` to `TextureAssignment`.
* Changed field `TextureAssignment.channel_index` to `TextureAssignment.channels`.
* Changed field `TextureAssignment.texture_scale` to `TextureAssignment.texture_transforms`.
* Renamed `Shader` to `ShaderProgram`.
* Changed the `ShaderProgram.texture` method to `ShaderProgram.textures` method that returns an optional list of assigned textures.
* Changed `load_model` and `load_map` functions to take `xc3_model_py.shader_database.ShaderDatabase` instead of a path.
* Changed `load_model_legacy` to take an optional `xc3_model_py.shader_database.ShaderDatabase` argument.

## 0.6.0 - 2024-06-09
### Added
* Added fields `unk_mesh_index1`, `ext_mesh_index`, and `base_mesh_index` to `Mesh`.
* Added method `mat_id` to `OutputAssignments`.
* Added method `update_weights` to `skinning.Weights`.
* Added method `add_influences` to `skinning.SkinWeights`.
* Added field `morph_blend_target` to `vertex.VertexBuffer` for base morph attributes.
* Added base morph attributes `Position2`, `Normal4`, `OldPosition`, and `Tangent2` to `vertex.AttributeType`.
* Added `SkinWeights2` and `BoneIndices2` to `vertex.AttributeType`.
* Added `LodData`, `LodItem`, and `LodGroup` types.

### Changed
* Adjusted the signature of `animation.Track.sample_` methods to also take the frame count.
* Adjusted fields to use references to make nested property access like `root.buffers.vertex_buffers` work as expected.
* Changed field `MorphTarget.tangent_deltas` to `MorphTarget.tangents` to better reflect in game data.
* Changed field `MorphTarget.normal_deltas` to `MorphTarget.normals` to better reflect in game data.
* Changed field `Models.base_lod_indices` to `Models.lod_data` to better reflect in game data.
* Renamed field `Mesh.lod` to `Mesh.lod_item_index` as an optional 0-based integer index.
* Changed the output file names for `save_images_rgba8` to always include the image index.

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