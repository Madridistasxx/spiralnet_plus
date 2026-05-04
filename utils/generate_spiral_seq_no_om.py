"""
OpenMesh-free spiral sequence generation.

Supports inputs such as:
  - PyTorch Geometric Data(x=[N,3], edge_index=[2,E], face=[3,F])
  - dict with keys: x/pos/v/vertices and face/edge_index
  - lightweight mesh-like object with .v/.f or .x/.face/.edge_index

The original OpenMesh version relied on:
  - mesh.vertices()
  - mesh.vv(vertex_handle)
  - mesh.points()
This rewrite builds vertex neighborhoods directly from faces or edge_index.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple, Any, Optional

import numpy as np
from sklearn.neighbors import KDTree


def _to_numpy(a: Any) -> np.ndarray:
    """Convert torch / numpy / list-like array to numpy."""
    if a is None:
        raise ValueError("Cannot convert None to numpy array.")
    if hasattr(a, "detach"):
        a = a.detach()
    if hasattr(a, "cpu"):
        a = a.cpu()
    if hasattr(a, "numpy"):
        return np.asarray(a.numpy())
    return np.asarray(a)


def _get_attr_or_key(obj: Any, names: Sequence[str]) -> Optional[Any]:
    """Read first existing attribute/key from an object or dict."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def mesh_to_arrays(mesh: Any) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract vertices, faces and edge_index from common mesh containers.

    Returns:
        vertices: [N, 3]
        faces: [F, 3] or None
        edge_index: [2, E] or None
    """
    vertices = _get_attr_or_key(mesh, ("v", "vertices", "pos", "x"))
    faces = _get_attr_or_key(mesh, ("f", "faces", "face"))
    edge_index = _get_attr_or_key(mesh, ("edge_index", "edges"))

    if vertices is None:
        raise ValueError("Could not find vertices. Expected one of: v, vertices, pos, x.")

    vertices = _to_numpy(vertices).astype(np.float64, copy=False)

    if faces is not None:
        faces = _to_numpy(faces).astype(np.int64, copy=False)
        # PyG stores faces as [3, F], while most mesh code uses [F, 3].
        if faces.ndim != 2:
            raise ValueError(f"faces must be 2D, got shape {faces.shape}.")
        if faces.shape[0] == 3 and faces.shape[1] != 3:
            faces = faces.T
        if faces.shape[1] != 3:
            raise ValueError(f"faces must have shape [F, 3] or [3, F], got {faces.shape}.")

    if edge_index is not None:
        edge_index = _to_numpy(edge_index).astype(np.int64, copy=False)
        if edge_index.ndim != 2:
            raise ValueError(f"edge_index must be 2D, got shape {edge_index.shape}.")
        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.T
        if edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E] or [E, 2], got {edge_index.shape}.")

    return vertices, faces, edge_index


def build_ordered_adjacency(
    num_vertices: int,
    faces: Optional[np.ndarray] = None,
    edge_index: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """Build deterministic ordered one-ring neighborhoods.

    Face-based construction preserves a local cyclic-ish order better than a
    plain edge list. If both faces and edge_index are provided, faces are used
    first, and any missing edge_index edges are appended.
    """
    neighbors: List[List[int]] = [[] for _ in range(num_vertices)]
    seen = [set() for _ in range(num_vertices)]

    def add_directed(i: int, j: int) -> None:
        if i == j:
            return
        if 0 <= i < num_vertices and 0 <= j < num_vertices and j not in seen[i]:
            neighbors[i].append(j)
            seen[i].add(j)

    if faces is not None:
        # For each oriented face (a,b,c), each vertex sees the other two in face order.
        for a, b, c in faces:
            add_directed(a, b)
            add_directed(a, c)
            add_directed(b, c)
            add_directed(b, a)
            add_directed(c, a)
            add_directed(c, b)

    if edge_index is not None:
        for i, j in edge_index.T:
            add_directed(int(i), int(j))
            add_directed(int(j), int(i))

    return neighbors


def _next_ring_from_adjacency(
    adjacency: Sequence[Sequence[int]],
    last_ring: Sequence[int],
    other: Sequence[int],
) -> List[int]:
    """OpenMesh-free equivalent of the original _next_ring logic."""
    res: List[int] = []
    last_set = set(last_ring)
    other_set = set(other)
    res_set = set()

    def is_new_vertex(idx: int) -> bool:
        return idx not in last_set and idx not in other_set and idx not in res_set

    for vh1 in last_ring:
        neigh = adjacency[int(vh1)]
        after_last_ring = False

        for vh2 in neigh:
            if after_last_ring and is_new_vertex(vh2):
                res.append(vh2)
                res_set.add(vh2)
            if vh2 in last_set:
                after_last_ring = True

        for vh2 in neigh:
            if vh2 in last_set:
                break
            if is_new_vertex(vh2):
                res.append(vh2)
                res_set.add(vh2)

    return res


def extract_spirals_from_arrays(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    seq_length: int,
    dilation: int = 1,
    edge_index: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """Generate spiral indices without OpenMesh.

    Args:
        vertices: [N, 3] vertex coordinates.
        faces: [F, 3] triangular faces. Can be None if edge_index is provided.
        seq_length: output sequence length per vertex.
        dilation: keep every dilation-th item from the longer spiral.
        edge_index: optional [2, E] edge index.

    Returns:
        Python list of shape approximately [N, seq_length].
    """
    vertices = np.asarray(vertices)
    if vertices.ndim != 2:
        raise ValueError(f"vertices must be 2D, got {vertices.shape}.")

    num_vertices = vertices.shape[0]
    if faces is None and edge_index is None:
        raise ValueError("Need either faces or edge_index to build neighborhoods.")

    adjacency = build_ordered_adjacency(num_vertices, faces, edge_index)
    kdt = KDTree(vertices, metric="euclidean")
    target_len = seq_length * dilation

    spirals: List[List[int]] = []
    for center in range(num_vertices):
        spiral = [center]
        one_ring = list(adjacency[center])
        last_ring = one_ring
        next_ring = _next_ring_from_adjacency(adjacency, last_ring, spiral)

        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < target_len:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring_from_adjacency(adjacency, last_ring, spiral)
            spiral.extend(last_ring)

        if len(next_ring) > 0:
            spiral.extend(next_ring)

        # Fallback/padding: if topology cannot provide enough vertices, use nearest vertices.
        if len(spiral) < target_len:
            k = min(target_len, num_vertices)
            nn = kdt.query(vertices[center:center + 1], k=k, return_distance=False).ravel().tolist()
            seen = set(spiral)
            spiral.extend([idx for idx in nn if idx not in seen])

        # If mesh has fewer vertices than requested, pad with the center vertex.
        if len(spiral) < target_len:
            spiral.extend([center] * (target_len - len(spiral)))

        spirals.append(spiral[:target_len][::dilation])

    return spirals


def extract_spirals(mesh: Any, seq_length: int, dilation: int = 1) -> List[List[int]]:
    """Drop-in style replacement for the OpenMesh version.

    Accepts PyG Data, dict, or objects with v/f/x/face/edge_index attributes.
    """
    vertices, faces, edge_index = mesh_to_arrays(mesh)
    return extract_spirals_from_arrays(
        vertices=vertices,
        faces=faces,
        edge_index=edge_index,
        seq_length=seq_length,
        dilation=dilation,
    )
