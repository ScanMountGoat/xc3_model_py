# xc3_model_py
Python bindings to [xc3_model](https://github.com/ScanMountGoat/xc3_lib) for high level and efficient data access to model files for Xenoblade 1 DE, Xenoblade 2, and Xenoblade 3.

## Introduction
Parsing and processing happens in optimized Rust code when calling `xc3_model_py.load_map` or `xc3_model_py.load_model`. All characters, models, and maps are converted to the same scene hierarchy representation. This avoids needing to add any special handling for maps vs characters. Pregenerated shader JSON databases are available from [xc3_lib](https://github.com/ScanMountGoat/xc3_lib/releases).

```python
import xc3_model_py

# Get a list of model roots.
roots = xc3_model_py.load_map("xenoblade3_dump/map/ma59a.wismhd", database_path="xc3.json")
for root in roots:
    for group in root.groups:
        for models in group.models:
            for material in models.materials:
                print(material.name)
                # The shader contains assignment information when specifying a JSON database.
                if material.shader is not None:
                    # Find the image texture and channel used for the ambient occlusion output.
                    ao = material.shader.sampler_channel_index(2, 'z')
                    if ao is not None:
                        sampler_index, channel_index = ao
                        image_texture_index = material.textures[sampler_index].image_texture_index
                        image_texture = root.image_textures[image_texture_index]
                        print(image_texture.name, 'xyzw'[channel_index])

            for model in models.models:
                # prints (num_instances, 4, 4)
                print(len(model.instances.shape))

# This returns only a single root.
root = xc3_model_py.load_model("xenoblade3_dump/chr/chr/01012013.wimdo", database_path="xc3.json")
for group in root.groups:
    for models in group.models:
        for model in models.models:
            # prints (1, 4, 4)
            print(len(model.instances.shape))

            # Access vertex and index data for this model.
            buffers = group.buffers[model.model_buffers_index]
            for buffer in buffers.vertex_buffers:
                for attribute in buffer.attributes:
                    print(attribute.attribute_type, attribute.data.shape)
```

Certain types like matrices and vertex atribute data are stored using `numpy.ndarray`. This greatly reduces conversion overhead and allows for more optimized Python code. xc3_model_py requires the numpy package to be installed. Blender already provides the numpy package, enabling the use of functions like `foreach_get` and `foreach_set` for efficient property access.

```python
# blender
blender_mesh.vertices.add(positions_array.shape[0])
blender_mesh.vertices.foreach_set('co', positions_array.reshape(-1))
```

## Documentation
See the [pyi stub file](https://github.com/ScanMountGoat/xc3_model_py/blob/main/xc3_model_py/__init__.pyi) for complete function and type information. This also enables autocomplete in supported editors like the Python extension for VSCode. The Python API attempts to match the Rust functions and types in xc3_model as closely as possible. 

## Installation
The compiled extension module can be imported just like any other Python file. On Windows, rename `xc3_model_py.dll` to `xc3_model_py.pyd`. If importing `xc3_model_py` fails, make sure the import path is specified correctly and the current Python version matches the version used when building. For installing in the current Python environment, install [maturin](https://github.com/PyO3/maturin) and use `maturin develop --release`.

## Building
Build the project with `cargo build --release`. This will compile a native python module for the current Python interpreter. For use with Blender, make sure to build for the Python version used by Blender. The easiest way to do this is to use the Python interpreter bundled with Blender. See the [PyO3 guide](https://pyo3.rs/main/building_and_distribution) for details. Some example commands are listed below for different operating systems. 

**Blender 3.6 on Windows**  
```
set PYO3_PYTHON = "C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin\python.exe"
cargo build --release
```

**Blender 3.6 on MacOS**  
```
PYO3_PYTHON="/Applications/Blender.app/Contents/Resources/3.6/python/bin/python3.10" cargo build --release
```

## Limitations
All data should be treated as immutable. Attempting to set fields will result in an error. Modifying list elements will appear to work, but changes will not be reflected when accessing the elements again. Types from `xc3_model_py` also cannot be constructed in any way from Python. These limitations may be lifted in the future. Write support may be added in the future as xc3_lib and xc3_model develop. 
