from typing import ClassVar, Optional, Tuple

class ShaderDatabase:
    @staticmethod
    def from_file(path: str) -> ShaderDatabase: ...

class ShaderProgram:
    output_dependencies: dict[str, int]
    outline_width: Optional[Dependency]
    normal_intensity: Optional[int]
    exprs: list[OutputExpr]

class OutputExpr:
    def func(self) -> Optional[OutputExprFunc]: ...
    def value(self) -> Optional[Dependency]: ...

class OutputExprFunc:
    op: Operation
    args: list[int]

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
    texcoords: list[int]

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
    AddNormalX: ClassVar[Operation]
    AddNormalY: ClassVar[Operation]
    Overlay: ClassVar[Operation]
    Overlay2: ClassVar[Operation]
    OverlayRatio: ClassVar[Operation]
    Power: ClassVar[Operation]
    Min: ClassVar[Operation]
    Max: ClassVar[Operation]
    Clamp: ClassVar[Operation]
    Abs: ClassVar[Operation]
    Fresnel: ClassVar[Operation]
    Sqrt: ClassVar[Operation]
    TexMatrix: ClassVar[Operation]
    TexParallaxX: ClassVar[Operation]
    TexParallaxY: ClassVar[Operation]
    ReflectX: ClassVar[Operation]
    ReflectY: ClassVar[Operation]
    ReflectZ: ClassVar[Operation]
    Unk: ClassVar[Operation]
