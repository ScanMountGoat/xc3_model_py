# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### unreleased
### Added
* Added method `model_space_transforms` to `Skeleton`.
* Added method `model_space_transforms` to `Animation`.
* Added method `skinning_transforms` to `Animation`.
* Added method `sample_transform` to `Track`.
* Added field `texcoord_name` to `ChannelAssignmentTexture`.
* Added field `sampler_index` to `Texture`.

### Changed
* Changed the `sample_translation`, `sample_rotation`, and `sample_scale` for `Track` to return `None` if empty.

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