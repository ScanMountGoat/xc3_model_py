from typing import ClassVar, Optional, Tuple

class ShaderDatabase:
    @staticmethod
    def from_file(path: str) -> ShaderDatabase: ...

class ShaderProgram:
    output_dependencies: dict[str, OutputExpr]
    outline_width: Optional[Dependency]
    normal_intensity: Optional[OutputExpr]

class OutputExpr:
    def func(self) -> Optional[OutputExprFunc]: ...
    def value(self) -> Optional[Dependency]: ...

class OutputExprFunc:
    op: Operation
    args: list[OutputExpr]

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

class Operation:
    Mix: ClassVar[Operation]
    Mul: ClassVar[Operation]
    Div: ClassVar[Operation]
    Add: ClassVar[Operation]
    Sub: ClassVar[Operation]
    Fma: ClassVar[Operation]
    MulRatio: ClassVar[Operation]
    AddNormal: ClassVar[Operation]
    Overlay: ClassVar[Operation]
    Overlay2: ClassVar[Operation]
    OverlayRatio: ClassVar[Operation]
    Power: ClassVar[Operation]
    Min: ClassVar[Operation]
    Max: ClassVar[Operation]
    Clamp: ClassVar[Operation]
    Abs: ClassVar[Operation]
    Fresnel: ClassVar[Operation]
    Unk: ClassVar[Operation]
