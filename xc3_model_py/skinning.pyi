from typing import Optional, ClassVar, Tuple
import numpy


class SkinWeights:
    bone_indices: numpy.ndarray
    weights: numpy.ndarray
    bone_names: list[str]

    def to_influences(
        self, weight_indices: numpy.ndarray) -> list[Influence]: ...


class Influence:
    bone_name: str
    weights: list[VertexWeight]


class VertexWeight:
    vertex_index: int
    weight: float
