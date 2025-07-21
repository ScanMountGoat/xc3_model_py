import numpy

class CollisionMeshes:
    vertices: numpy.ndarray
    meshes: list[CollisionMesh]

class CollisionMesh:
    name: str
    instances: numpy.ndarray
    indices: numpy.ndarray
