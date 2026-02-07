# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## unreleased
### Added
* Added support for wimdo export for Xenoblade Chronicles X Definitive Edition.
* Added method `material.OutputAssignment.merge_xyz` and associated types for combining XYZ assignments if possible.
* Added additional variants to `shader_database.Operation`.
* Added `shader_database.Dependency.int` variant.
* Added field `unk` to `LodData` to better preserve in game data.
* Added additional variants to `material.RenderPassType`.
* Added field `alt_textures` to `material.Material`.
* Added field `lod_bias` to `Sampler`.
* Added additional variants to `material.StencilValue` and `material.StencilMode`.
* Added additional variants to `TextureUsage`.
* Added field `sampler_index2` to `material.Texture`.
* Added method `alpha_texture_channel_index` to `material.Material`.
* Added type `material.MaterialFlags`.

### Changed
* Changed numpy arrays to be row-major to properly match conventions.
* Moved `bone_names` field for `skinning.SkinWeights` to parameters for `to_influences` and `add_influences` methods.
* Renamed `shader_database.Dependency.constant` to `shader_database.Dependency.float`.
* Renamed field `material.Material.m_unks2_2` to `m_unks2`.
* Changed field `material.Material.alpha_test` to use type `material.Texture` to better match in game data.
* Changed field `material.Material.flags` to use type `material.MaterialFlags` to better match in game data.

### Removed
* Removed field `unk5` from `LodItem`.
* Removed type `material.TextureAlphaTest`.

### Fixed
* Fixed an issue where fields on nested types for `material.Material` were not mutable.

## 0.15.0 - 2025-06-09
### Added
* Added many additional fields to `material.MaterialParameters`.
* Added additional variants to `shader_database.Operation` and split vector operations into separate operations for XYZ channels.

### Changed
* Renamed `material.ChannelAssignment` and associated types to `material.AssignmentValue`.
* Combined `shader_database.OutputDependencies` and `shader_database.TextureLayer` into `shader_database.OutputExpr`.
* Renamed `shader_database.BlendMode` to `shader_database.Operation` with many additional variants.
* Renamed method `material.AssignmentValue.value` to `material.AssignmentValue.float`.
* Changed field `shader_database.AssignmentValueAttribute.channel_index` to `shader_database.AssignmentValueAttribute.channel`.
* Changed field `shader_database.TextureDependency.texcoords` to be assignment indices.
* Replaced fields `texcoord_name` and `texcoord_transforms` fields on `material.TextureAssignment` with `material.texcoords`.
* Changed indices for `shader_database.OutputAssignment` to be `None` if nothing is assigned.
* Changed fields `output_dependencies` and `normal_intensity` for `shader_database.ShaderProgram` to index into the `exprs` field.

### Removed
* Removed types `shader_database.TexCoord` and `shader_database.TexCoordParams`.

## 0.14.0 - 2025-03-04
### Added
* Added support for wimdo models from Xenoblade Chronicles X Definitive Edition.
* Added fields `material_color` and `ava_skin` to `material.MaterialParameters`.
* Added additional enum variants to `TextureUsage`, `material.StencilMode`, and `material.StencilValue`.
* Added field `sampler_index` to `material.TextureAlphaTest`.

### Changed
* Changed field `material.Material.alpha_test_ref` to type `float`.
* Renamed field `material.Material.m_unks3_1` to `gbuffer_flags`.

## 0.13.0 - 2025-03-06
### Added
* Added `animation.Animation.fcurves` and `animation.FCurves` for computing fcurves compatible with Blender.
* Added `Transform` type for decomposed translation, rotation, scale (TRS) transforms.
* Added types `vertex.UnkBuffer` and `vertex.UnkDataBuffer`.
* Added fields `unk_buffers` and `unk_data_buffer` to `vertex.ModelBuffers`.

### Changed
* Changed `animation.Track.sample_transform` to return `Transform`.
* Changed the type of `Bone.transform` to `Transform`.

## 0.12.0 - 2025-01-15
### Added
* Added field `root_translation` to `animation.Animation` for additional translation of the root bone.
* Added variant `Unk` to `material.ColorWriteMode`.
* Added variant `Unk21` to `TextureUsage`.
* Added function `load_collisions` for loading collision data from `.wiidcm` or `.idcm` files.
* Added `collision` module for collision mesh data.

### Changed
* Changed animation loading methods like `animation.Animation.model_space_transforms` to include root bone translation if present.

### Removed
* Removed `shader_database.ModelPrograms` and `shader_database.MapPrograms` as they are no longer used.
* Removed `map` and `model` methods from `shader_database.ShaderDatabase` to match database changes. Access the shader from the material instead.

## 0.11.0 - 2024-11-27
### Added
* Added field `outline_width` to `material.OutputAssignments`.
* Added field `outline_width` to `shader_database.ShaderProgram`.
* Added field `skinning` to `Models`.
* Added types `Skinning`, `Bone`, `BoneBounds`, `BoneConstraint`, and `BoneConstraintType` to `skinning` module.
* Added method `monolib.ShaderTextures.global_textures` for accessing all supported  sampler names and textures.

### Changed
* Changed `shader_database.BufferDependency.index` to be an `int` or `None`.
* Changed `channels` fields for all types in `shader_database` to `channel` with a single character string or `None`.
* Changed shader database files to use a custom binary format.

## 0.10.0 - 2024-10-31
### Added
* Added type `monolib.ShaderTextures`.
* Added type `material.FurShellParams`.
* Added type `material.WorkCallback` for the element type of `Material.work_callbacks`.
* Added variant `Overlay` to `shader_database.LayerBlendMode`.
* Added type `shader_database.OutputDependencies` to store dependencies and layers.
* Added function `decode_images_png`.

### Changed
* Moved material types in `materials` module.
* Changed `material.OutputAssignment` to store layers per channel.
* Changed `material.OutputLayerAssignment` to `material.LayerChannelAssignment`.

### Removed
* Removed fields `color_layers` and `normal_layers` from `shader_database.ShaderProgram`.
* Removed field `alpha_test_ref` from `material.MaterialParameters`.

## 0.9.0 - 2024-09-18
### Added
* Added field `normal_layers` to `shader_database.ShaderProgram`.
* Added field `layers` to `OutputAssignment`.
* Added type `OutputLayerAssignment`.
* Added field `color_layers` to `shader_database.ShaderProgram`.
* Added fields `blend_mode` and `is_fresnel` to `OutputLayerAssignment`.
* Added fields `blend_mode` and `is_fresnel` to `shader_database.TextureLayer`.
* Added type `shader_database.LayerBlendMode`.

### Changed
* Changed method `ChannelAssignment.textures` to `ChannelAssignment.texture` with a single texture assignment.
* Combined fields `name` and `channel` for `shader_database.TextureLayer` to a single field `value` of type `shader_database.Dependency`.

## 0.8.0 - 2024-08-16
### Added
* Added field `outline_buffers` to `vertex.ModelBuffers`.
* Added type `vertex.OutlineBuffer`.
* Added field `color` to `Material`.
* Added field `primitive_type` to `vertex.IndexBuffer`.
* Added type `vertex.PrimitiveType`.
* Added variants `WeightIndex2`, `Unk15`, `Unk16`, `Unk18`, `Unk24`, `Unk25`, `Unk26`, `Unk30`, `Unk31`, `Normal2`, `ValInf`, `Normal3`, `VertexColor3`, and `Flow` to `vertex.AttributeType`.
* Added type `EncodeSurfaceRgbaFloat32Args` for encoding floating point images.
* Added type `EncodeSurfaceRgba8Args` for encoding images.
* Added function `encode_images_rgbaf32` for encoding floating point images in parallel.
* Added function `encode_images_rgba8` for encoding images in parallel.
* Added function `decode_images_rgbaf32` for decoding images in parallel.
* Added type `Dds` as an opaque wrapper for DDS files.
* Added static method `ImageTexture.from_dds` for low cost conversion of DDS files.

### Changed
* Improved accuracy of vertex data rebuilding.

### Removed
* Removed field `mat_color` from `MaterialParameters`.
* Removed method `ModelRoot.decode_images_rgbaf32`. Use `decode_images_rgbaf32` instead.
* Removed method `MapRoot.decode_images_rgbaf32`. Use `decode_images_rgbaf32` instead.

### Fixed
* Fixed an issue where `ModelRoot.save_images_rgba8` and `MapRoot.save_images_rgba8` did not work due to incorrect feature flags.

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