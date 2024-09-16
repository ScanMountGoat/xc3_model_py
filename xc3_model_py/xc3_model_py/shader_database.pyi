from typing import ClassVar, Optional, Tuple

class ShaderDatabase:
    @staticmethod
    def from_file(path: str) -> ShaderDatabase: ...
    def model(self, name: str) -> Optional[ModelPrograms]: ...
    def map(self, name: str) -> Optional[MapPrograms]: ...

class ModelPrograms:
    programs: list[ShaderProgram]

class MapPrograms:
    map_models: list[ModelPrograms]
    prop_models: list[ModelPrograms]
    env_models: list[ModelPrograms]

class ShaderProgram:
    output_dependencies: dict[str, list[Dependency]]
    color_layers: list[TextureLayer]
    normal_layers: list[TextureLayer]

class Dependency:
    def constant(self) -> Optional[float]: ...
    def buffer(self) -> Optional[BufferDependency]: ...
    def texture(self) -> Optional[TextureDependency]: ...
    def attribute(self) -> Optional[AttributeDependency]: ...

class BufferDependency:
    name: str
    field: str
    index: int
    channels: str

class TextureDependency:
    name: str
    channels: str
    texcoords: list[TexCoord]

class TexCoord:
    name: str
    channels: str
    params: Optional[TexCoordParams]

class TexCoordParams:
    def scale(self) -> Optional[BufferDependency]: ...
    def matrix(
        self,
    ) -> Optional[
        Tuple[BufferDependency, BufferDependency, BufferDependency, BufferDependency]
    ]: ...

class AttributeDependency:
    name: str
    channels: str

class TextureLayer:
    value: Dependency
    ratio: Optional[Dependency]
    blend_mode: LayerBlendMode
    is_fresnel: bool

class LayerBlendMode:
    Mix: ClassVar[LayerBlendMode]
    MixRatio: ClassVar[LayerBlendMode]
    Add: ClassVar[LayerBlendMode]
    AddNormal: ClassVar[LayerBlendMode]
