import builtins
from typing import ClassVar, Optional, Tuple

class ShaderDatabase:
    @staticmethod
    def from_file(path: str) -> ShaderDatabase: ...

class ShaderProgram:
    output_dependencies: dict[str, int]
    outline_width: Optional[Value]
    normal_intensity: Optional[int]
    val_inf_intensity: Optional[int]
    discard_condition: Optional[int]
    exprs: list[OutputExpr]

class OutputExpr:
    def func(self) -> Optional[OutputExprFunc]: ...
    def value(self) -> Optional[Value]: ...

class OutputExprFunc:
    op: Operation
    args: list[int]

class Value:
    def int(self) -> Optional[builtins.int]: ...
    def float(self) -> Optional[builtins.float]: ...
    def parameter(self) -> Optional[Parameter]: ...
    def texture(self) -> Optional[Texture]: ...
    def attribute(self) -> Optional[Attribute]: ...

class Parameter:
    name: str
    field: str
    index: Optional[int]
    channel: Optional[str]

    def __init__(
        self,
        name: str,
        field: str,
        index: Optional[int],
        channel: Optional[str],
    ) -> None: ...

class Texture:
    name: str
    channel: Optional[str]
    texcoords: list[int]

class Attribute:
    name: str
    channel: Optional[str]

class Operation:
    Unk: ClassVar[Operation]
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
    Floor: ClassVar[Operation]
    Select: ClassVar[Operation]
    Equal: ClassVar[Operation]
    NotEqual: ClassVar[Operation]
    Less: ClassVar[Operation]
    Greater: ClassVar[Operation]
    LessEqual: ClassVar[Operation]
    GreaterEqual: ClassVar[Operation]
    Dot4: ClassVar[Operation]
    NormalMapX: ClassVar[Operation]
    NormalMapY: ClassVar[Operation]
    NormalMapZ: ClassVar[Operation]
    MonochromeX: ClassVar[Operation]
    MonochromeY: ClassVar[Operation]
    MonochromeZ: ClassVar[Operation]
    Negate: ClassVar[Operation]
    FurInstanceAlpha: ClassVar[Operation]
    Float: ClassVar[Operation]
    Int: ClassVar[Operation]
    Uint: ClassVar[Operation]
    Truncate: ClassVar[Operation]
    FloatBitsToInt: ClassVar[Operation]
    IntBitsToFloat: ClassVar[Operation]
    UintBitsToFloat: ClassVar[Operation]
    InverseSqrt: ClassVar[Operation]
    Not: ClassVar[Operation]
    LeftShift: ClassVar[Operation]
    RightShift: ClassVar[Operation]
    PartialDerivativeX: ClassVar[Operation]
    PartialDerivativeY: ClassVar[Operation]
    Exp2: ClassVar[Operation]
    Log2: ClassVar[Operation]
    Sin: ClassVar[Operation]
    Cos: ClassVar[Operation]

class OutputExprXyz:
    def func(self) -> Optional[OutputExprFuncXyz]: ...
    def value(self) -> Optional[ValueXyz]: ...

class ValueXyz:
    def texture(self) -> Optional[TextureXyz]: ...
    def float(
        self,
    ) -> Optional[Tuple[builtins.float, builtins.float, builtins.float]]: ...
    def attribute(self) -> Optional[AttributeXyz]: ...
    def parameter(self) -> Optional[ParameterXyz]: ...

class OutputExprFuncXyz:
    op: OperationXyz
    args: list[int]
    channel: Optional[ChannelXyz]

class OperationXyz:
    Unk: ClassVar[OperationXyz]
    Mix: ClassVar[OperationXyz]
    Mul: ClassVar[OperationXyz]
    Div: ClassVar[OperationXyz]
    Add: ClassVar[OperationXyz]
    Sub: ClassVar[OperationXyz]
    Fma: ClassVar[OperationXyz]
    MulRatio: ClassVar[OperationXyz]
    Overlay: ClassVar[OperationXyz]
    Overlay2: ClassVar[OperationXyz]
    OverlayRatio: ClassVar[OperationXyz]
    Power: ClassVar[OperationXyz]
    Min: ClassVar[OperationXyz]
    Max: ClassVar[OperationXyz]
    Clamp: ClassVar[OperationXyz]
    Abs: ClassVar[OperationXyz]
    Fresnel: ClassVar[OperationXyz]
    Sqrt: ClassVar[OperationXyz]
    Reflect: ClassVar[OperationXyz]
    Floor: ClassVar[OperationXyz]
    Select: ClassVar[OperationXyz]
    Equal: ClassVar[OperationXyz]
    NotEqual: ClassVar[OperationXyz]
    Less: ClassVar[OperationXyz]
    Greater: ClassVar[OperationXyz]
    LessEqual: ClassVar[OperationXyz]
    GreaterEqual: ClassVar[OperationXyz]
    Monochrome: ClassVar[OperationXyz]
    Negate: ClassVar[OperationXyz]
    Float: ClassVar[OperationXyz]
    Int: ClassVar[OperationXyz]
    Uint: ClassVar[OperationXyz]
    Truncate: ClassVar[OperationXyz]
    FloatBitsToInt: ClassVar[OperationXyz]
    IntBitsToFloat: ClassVar[OperationXyz]
    UintBitsToFloat: ClassVar[OperationXyz]
    InverseSqrt: ClassVar[OperationXyz]
    Not: ClassVar[OperationXyz]
    LeftShift: ClassVar[OperationXyz]
    RightShift: ClassVar[OperationXyz]
    Exp2: ClassVar[OperationXyz]
    Log2: ClassVar[OperationXyz]
    Sin: ClassVar[OperationXyz]
    Cos: ClassVar[OperationXyz]

class TextureXyz:
    name: str
    channel: Optional[ChannelXyz]
    texcoords: list[int]

class AttributeXyz:
    name: str
    channel: Optional[ChannelXyz]

class ParameterXyz:
    name: str
    field: str
    index: Optional[int]
    channel: Optional[ChannelXyz]

class ChannelXyz:
    Xyz: ClassVar[ChannelXyz]
    X: ClassVar[ChannelXyz]
    Y: ClassVar[ChannelXyz]
    Z: ClassVar[ChannelXyz]
    W: ClassVar[ChannelXyz]
