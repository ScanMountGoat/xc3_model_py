from typing import Optional, ClassVar, Tuple
import numpy


class SkinWeights:
    bone_indices: numpy.ndarray
    weights: numpy.ndarray
    bone_names: list[str]

    def __init__(self, bone_indices: numpy.ndarray,
                 weights: numpy.ndarray, bone_names: list[str]) -> None: ...

    def to_influences(
        self, weight_indices: numpy.ndarray) -> list[Influence]: ...


class Influence:
    bone_name: str
    weights: list[VertexWeight]

    def __init__(self, bone_name: str,
                 weights: list[VertexWeight]) -> None: ...


class VertexWeight:
    vertex_index: int
    weight: float

    def __init__(self, vertex_index: int, weight: float) -> None: ...
