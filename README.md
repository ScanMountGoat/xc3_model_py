# xc3_model_py [![PyPI](https://img.shields.io/pypi/v/xc3_model_py)](https://pypi.org/project/xc3_model_py/)
Python bindings to [xc3_model](https://github.com/ScanMountGoat/xc3_lib) for high level and efficient data access to model files for Xenoblade 1 DE, Xenoblade 2, and Xenoblade 3.

## Installation
The package can be installed for Python version 3.9, 3.10, 3.11, or 3.12 using pip on newer versions of Windows, Linux, or MacOS. The prebuilt wheels (.whl files) are included only for situations where pip might not be available such as for plugin development for applications. The wheels are zip archives and can be extracted to obtain the compiled .pyd or .so file. xc3_model_py requires the `numpy` package for transforms and vertex data.

Installing: `pip install xc3_model_py`  
Updating: `pip install xc3_model_py --upgrade`

## Introduction
Parsing and processing happens in optimized Rust code when calling `xc3_model_py.load_map` or `xc3_model_py.load_model`. All characters, models, and maps are converted to the same scene hierarchy representation. This avoids needing to add any special handling for maps vs characters. Pregenerated shader JSON databases are available from [xc3_lib](https://github.com/ScanMountGoat/xc3_lib/releases). For more advanced usage, see [xenoblade_blender](https://github.com/ScanMountGoat/xenoblade_blender).

```python
import xc3_model_py

# Get a list of MapRoot.
roots = xc3_model_py.load_map("xenoblade3_dump/map/ma59a.wismhd", database_path="xc3.json")
for root in roots:
    for group in root.groups:
        for models in group.models:
            for material in models.materials:
                print(material.name)
                # The shader contains assignment information when specifying a JSON database.

            for model in models.models:
                buffers = group.buffers[model.model_buffers_index]

                # prints (num_instances, 4, 4)
                print(len(model.instances.shape))
```

```python
# This returns only a single ModelRoot.
root = xc3_model_py.load_model("xenoblade3_dump/chr/chr/01012013.wimdo", database_path="xc3.json")
for material in root.models.materials:
    print(material.name)

for model in root.models.models:
    # prints (1, 4, 4)
    print(len(model.instances.shape))

    # Access vertex and index data for this model.
    buffers = root.buffers
    for buffer in buffers.vertex_buffers:
        for attribute in buffer.attributes:
            print(attribute.attribute_type, attribute.data.shape)

    # Access vertex skinning data for each mesh.
    for mesh in model.meshes:
        vertex_buffer = buffers.vertex_buffers[mesh.vertex_buffer_index]

        if buffers.weights is not None:
            # Calculate the index offset based on the weight group for this mesh.
            pass_type = root.models.materials[mesh.material_index].pass_type
            start_index = buffers.weights.weights_start_index(mesh.flags2, mesh.lod, pass_type)

            weight_buffer = buffers.weights.weight_buffer(mesh.flags2)
            if weight_buffer is not None:
                # Get vertex skinning attributes.
                for attribute in vertex_buffer.attributes:
                    if attribute.attribute_type == xc3_model_py.vertex.AttributeType.WeightIndex:
                        # Find the actual per vertex skinning information.
                        weight_indices = attribute.data[:, 0] + start_index
                        skin_weights = weight_buffer.weights[weight_indices]
                        # Note that these indices index into a different bone list than the skeleton.
                        bone_indices = weight_buffer.bone_indices[weight_indices, 0]
                        bone_name = weight_buffer.bone_names[bone_indices[0]]
```

Certain types like matrices and vertex atribute data are stored using `numpy.ndarray`. All transformation matrices are column-major to match the Rust code in xc3_model. This greatly reduces conversion overhead and allows for more optimized Python code. xc3_model_py requires the numpy package to be installed. Blender already provides the numpy package, enabling the use of functions like `foreach_get` and `foreach_set` for efficient property access.

```python
# blender
blender_mesh.vertices.add(positions_array.shape[0])
blender_mesh.vertices.foreach_set('co', positions_array.reshape(-1))
```

Animations can be loaded from a file all at once. The track type is currently opaque, meaning that implementation details are not exposed. The values can be sampled at the desired frame using the appropriate methods.

```python
import xc3_model_py

path = "xenoblade3_dump/chr/ch/ch01027000_event.mot"
animations = xc3_model_py.load_animations(path)

for animation in animations:
    print(animation.name, animation.space_mode, animation.play_mode, animation.blend_mode)
    print(f'frames: {animation.frame_count}, tracks: {len(animation.tracks)}')

    track = animation.tracks[0]

    # Each track references a bone in one of three ways.
    bone_index = track.bone_index()
    bone_hash = track.bone_hash()
    bone_name = track.bone_name()
    if bone_index is not None:
        pass
    elif bone_hash is not None:
        # Use xc3_model_py.murmur3(bone_name) for hashing the skeleton bones.
        pass
    elif bone_name is not None:
        pass

    # Sample the transform for a given track at each frame.
    # This essentially "bakes" the keyframes of the animation.
    for frame in range(animation.frame_count:)
        print(track.sample_scale(frame, animation.frame_count))
        print(track.sample_rotation(frame, animation.frame_count))
        print(track.sample_translation(frame, animation.frame_count))
    print()
```

xc3_model_py enables Rust log output by default to use with Python's `logging` module.
Logging can be disabled entirely if not needed using `logging.disable()`.

```python
import logging

# Configure log messages to include more information.
FORMAT = '%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
```

## Documentation
See the [pyi stub file](https://github.com/ScanMountGoat/xc3_model_py/blob/main/xc3_model_py/__init__.pyi) for complete function and type information. This also enables autocomplete in supported editors like the Python extension for VSCode. The Python API attempts to match the Rust functions and types in xc3_model as closely as possible. 

## Installation
The compiled extension module can be imported just like any other Python file. On Windows, rename `xc3_model_py.dll` to `xc3_model_py.pyd`. If importing `xc3_model_py` fails, make sure the import path is specified correctly and the current Python version matches the version used when building. For installing in the current Python environment, install [maturin](https://github.com/PyO3/maturin) and use `maturin develop --release`.

## Building
Build the project with `cargo build --release`. This will compile a native python module for the current Python interpreter. For use with Blender, make sure to build for the Python version used by Blender. This can be achieved by activating a virtual environment with the appropriate Python version or setting the Python interpeter using the `PYO3_PYTHON` environment variable. See the [PyO3 guide](https://pyo3.rs) for details.

## Limitations
Some types from `xc3_model_py` are opaque wrappers around the underlying Rust types and cannot be constructed in any way from Python. Some of these limitations should hopefully be resolved in the future.
