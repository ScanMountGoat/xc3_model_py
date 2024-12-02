from typing import ClassVar, Optional, Tuple

class ShaderDatabase:
    @staticmethod
    def from_file(path: str) -> ShaderDatabase: ...

class ShaderProgram:
    output_dependencies: dict[str, list[OutputDependencies]]
    outline_width: Optional[Dependency]

class OutputDependencies:
    dependencies: list[Dependency]
    layers: list[TextureLayer]

class Dependency:
    def constant(self) -> Optional[float]: ...
    def buffer(self) -> Optional[BufferDependency]: ...
    def texture(self) -> Optional[TextureDependency]: ...
    def attribute(self) -> Optional[AttributeDependency]: ...

class BufferDependency:
    name: str
    field: str
    index: Optional[int]
    channel: Optional[str]

class TextureDependency:
    name: str
    channel: Optional[str]
    texcoords: list[TexCoord]

class TexCoord:
    name: str
    channel: Optional[str]
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
    channel: Optional[str]

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
    Overlay: ClassVar[LayerBlendMode]
