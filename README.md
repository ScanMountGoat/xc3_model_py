# xc3_model_py
Python bindings to [xc3_model](https://github.com/ScanMountGoat/xc3_lib). xc3_model_py provides high level and efficient data access to model files for Xenoblade 1 DE, Xenoblade 2, and Xenblade 3.

## Introduction
xc3_model_py is designed for simplicity and speed. All of the processing happens in optimized Rust code when calling `xc3_model_py.load_map` or `xc3_model_py.load_model`. All characters, models, and maps are converted to the same scene hierarchy representation. This avoids needing to add any special handling for maps vs characters. 

```python
import xc3_model_py

# Get a list of model roots.
roots = xc3_model_py.load_map("xenoblade3_dump/map/ma59a.wismhd")
for root in roots:
    for group in root.groups:
        for material in group.materials:
            print(material.name)
        for model in group.models:
            # prints (num_instances, 4, 4)
            print(len(model.instances.shape))

# This returns only a single root.
root = xc3_model_py.load_model("xenoblade3_dump/chr/chr/01012013.wimdo")
for group in root.groups:
    for model in group.models:
        # prints (1, 4, 4)
        print(len(model.instances.shape))
```

Certain types like matrices and vertex atribute data are stored using `numpy.ndarray`. This allows loading times to be close pure Rust code and allows for more optimized Python code. xc3_model_py requires the numpy package to be installed. Blender already provides the numpy package. Blender addons can also use functions `foreach_get` and `foreach_set` for very efficiently propert access.

```python
# blender
blender_mesh.vertices.add(positions_array.shape[0])
blender_mesh.vertices.foreach_set('co', positions_array.reshape(-1))
```

## Importing
The compiled extension module can be imported just like any other Python file. On Windows, rename `xc3_model_py.dll` to `xc3_model_py.pyd`. If importing `xc3_model_py` fails, make sure the import path is specified correctly and the current Python version matches the version used when building. 

## Building
Build the project with `cargo build --release`. This will compile a native python module for the current Python interpreter. For use with Blender, make sure to build for the Python version used by Blender. The easiest way to do this is to use the Python interpreter bundled with Blender. See the [PyO3 guide](https://pyo3.rs/main/building_and_distribution) for details. Some example commands are listed below for different operating systems. 

**Blender 3.6 on Windows**
```
set PYO3_PYTHON = "C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin\python.exe"
cargo build --release
```

**Blender 3.6 on MacOS**
`PYO3_PYTHON="/Applications/Blender.app/Contents/Resources/3.6/python/bin/python3.10" cargo build --release`