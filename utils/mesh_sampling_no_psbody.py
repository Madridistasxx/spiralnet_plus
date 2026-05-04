"""Mesh sampling utilities without a dependency on psbody.mesh.

This module keeps the public API of the original mesh_sampling.py as much as
possible, but replaces psbody.mesh.Mesh and its AABB-tree nearest-point query
with small NumPy/SciPy based implementations.

Expected mesh object interface:
    mesh.v: (N, 3) float array of vertices
    mesh.f: (F, 3) int array of triangular faces

You may pass either SimpleMesh or any object exposing compatible .v and .f
attributes.
"""

import heapq
import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class SimpleMesh:
    """Minimal replacement for psbody.mesh.Mesh used by this file."""

    v: np.ndarray
    f: np.ndarray

    def __post_init__(self):
        self.v = np.asarray(self.v, dtype=np.float64)
        self.f = np.asarray(self.f, dtype=np.int64)
        if self.v.ndim != 2 or self.v.shape[1] != 3:
            raise ValueError("v must have shape (N, 3)")
        if self.f.ndim != 2 or self.f.shape[1] != 3:
            raise ValueError("f must have shape (F, 3)")

    def compute_aabb_tree(self):
        """Compatibility shim for the subset of psbody.mesh used here.

        This is not a real AABB tree; it performs exact vectorized nearest-point
        queries over all triangles. It is dependency-free and accurate, but can
        be slower than psbody's compiled AABB tree on large meshes.
        """
        return _TriangleNearestQuery(self)


# Backwards-compatible alias: code below can still call Mesh(v=..., f=...).
Mesh = SimpleMesh


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def get_vert_connectivity(mesh_v, mesh_f):
    """Return sparse vertex-vertex connectivity for a triangular mesh."""
    mesh_v = np.asarray(mesh_v)
    mesh_f = np.asarray(mesh_f, dtype=np.int64)
    vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))

    for i in range(3):
        IS = mesh_f[:, i]
        JS = mesh_f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.ravel()), row(JS.ravel())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def get_vertices_per_edge(mesh_v, mesh_f):
    """Return an (E, 2) array containing each undirected edge once."""
    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:, 0] < result[:, 1]]
    return result.astype(np.int64)


def _closest_points_on_triangles(point, triangles):
    """Closest points from one 3D point to many triangles.

    Parameters
    ----------
    point : (3,) array
    triangles : (F, 3, 3) array

    Returns
    -------
    closest : (F, 3) array
    bary : (F, 3) array
        Barycentric weights of each closest point relative to triangle vertices.
    """
    p = np.asarray(point, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.float64)
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]

    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    closest = np.empty_like(a)
    bary = np.zeros((tri.shape[0], 3), dtype=np.float64)
    unresolved = np.ones(tri.shape[0], dtype=bool)

    # Vertex A region
    mask = (d1 <= 0.0) & (d2 <= 0.0)
    closest[mask] = a[mask]
    bary[mask, 0] = 1.0
    unresolved &= ~mask

    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)

    # Vertex B region
    mask = unresolved & (d3 >= 0.0) & (d4 <= d3)
    closest[mask] = b[mask]
    bary[mask, 1] = 1.0
    unresolved &= ~mask

    # Edge AB region
    vc = d1 * d4 - d3 * d2
    mask = unresolved & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denom = d1[mask] - d3[mask]
    v = np.divide(d1[mask], denom, out=np.zeros_like(denom), where=denom != 0)
    closest[mask] = a[mask] + v[:, None] * ab[mask]
    bary[mask, 0] = 1.0 - v
    bary[mask, 1] = v
    unresolved &= ~mask

    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    # Vertex C region
    mask = unresolved & (d6 >= 0.0) & (d5 <= d6)
    closest[mask] = c[mask]
    bary[mask, 2] = 1.0
    unresolved &= ~mask

    # Edge AC region
    vb = d5 * d2 - d1 * d6
    mask = unresolved & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denom = d2[mask] - d6[mask]
    w = np.divide(d2[mask], denom, out=np.zeros_like(denom), where=denom != 0)
    closest[mask] = a[mask] + w[:, None] * ac[mask]
    bary[mask, 0] = 1.0 - w
    bary[mask, 2] = w
    unresolved &= ~mask

    # Edge BC region
    va = d3 * d6 - d5 * d4
    mask = unresolved & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    denom = (d4[mask] - d3[mask]) + (d5[mask] - d6[mask])
    w = np.divide(d4[mask] - d3[mask], denom, out=np.zeros_like(denom), where=denom != 0)
    closest[mask] = b[mask] + w[:, None] * (c[mask] - b[mask])
    bary[mask, 1] = 1.0 - w
    bary[mask, 2] = w
    unresolved &= ~mask

    # Face interior region
    mask = unresolved
    denom = va[mask] + vb[mask] + vc[mask]
    v = np.divide(vb[mask], denom, out=np.zeros_like(denom), where=denom != 0)
    w = np.divide(vc[mask], denom, out=np.zeros_like(denom), where=denom != 0)
    u = 1.0 - v - w
    closest[mask] = u[:, None] * a[mask] + v[:, None] * b[mask] + w[:, None] * c[mask]
    bary[mask, 0] = u
    bary[mask, 1] = v
    bary[mask, 2] = w

    return closest, bary


def _nearest_part_from_barycentric(bary, tol=1e-8):
    """Map barycentric coordinates to psbody-like nearest part IDs.

    0 = face interior, 1 = edge v0-v1, 2 = edge v1-v2, 3 = edge v2-v0,
    4 = vertex v0, 5 = vertex v1, 6 = vertex v2.
    """
    zero = np.abs(bary) <= tol
    one = np.abs(bary - 1.0) <= tol

    if one[0]:
        return 4
    if one[1]:
        return 5
    if one[2]:
        return 6
    if zero[2]:
        return 1
    if zero[0]:
        return 2
    if zero[1]:
        return 3
    return 0


class _TriangleNearestQuery:
    """Exact nearest-point queries over all triangles of a SimpleMesh."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.triangles = mesh.v[mesh.f]

    def nearest(self, points, nearest_part=True):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if len(self.triangles) == 0:
            raise ValueError("source mesh has no faces")

        nearest_faces = np.empty((points.shape[0], 1), dtype=np.int64)
        nearest_parts = np.empty((points.shape[0], 1), dtype=np.int64)
        nearest_vertices = np.empty((points.shape[0], 3), dtype=np.float64)

        for i, point in enumerate(points):
            closest, bary = _closest_points_on_triangles(point, self.triangles)
            dist2 = np.einsum("ij,ij->i", closest - point, closest - point)
            face_id = int(np.argmin(dist2))
            nearest_faces[i, 0] = face_id
            nearest_vertices[i] = closest[face_id]
            nearest_parts[i, 0] = _nearest_part_from_barycentric(bary[face_id])

        return nearest_faces, nearest_parts, nearest_vertices


def vertex_quadrics(mesh):
    """Compute a 4x4 quadric error matrix for each vertex."""
    v_quadrics = np.zeros((len(mesh.v), 4, 4), dtype=np.float64)

    for f_idx in range(len(mesh.f)):
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.ones((3, 1))))
        _, _, vh = np.linalg.svd(verts)
        eq = vh[-1, :].reshape(-1, 1)
        normal_norm = np.linalg.norm(eq[0:3])
        if normal_norm == 0:
            continue
        eq = eq / normal_norm

        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics


def setup_deformation_transfer(source, target, use_normals=False):
    """Build a sparse interpolation matrix from source vertices to target vertices.

    The original psbody implementation uses an AABB tree. This version uses the
    SimpleMesh compatibility query above. `use_normals` is kept for API
    compatibility and is currently unused, matching the original file behavior.
    """
    rows = np.zeros(3 * target.v.shape[0], dtype=np.int64)
    cols = np.zeros(3 * target.v.shape[0], dtype=np.int64)
    coeffs_v = np.zeros(3 * target.v.shape[0], dtype=np.float64)

    nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree().nearest(target.v, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.reshape((-1, 3))

    for i in range(target.v.shape[0]):
        f_id = nearest_faces[i]
        nearest_f = source.f[f_id]
        nearest_v = nearest_vertices[i]

        rows[3 * i:3 * i + 3] = i
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        if n_id == 0:
            # Closest point is inside the triangle: solve barycentric-like weights.
            A = np.vstack((source.v[nearest_f])).T
            coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v, rcond=None)[0]
        elif 1 <= n_id <= 3:
            # Closest point is on an edge.
            v0 = nearest_f[n_id - 1]
            v1 = nearest_f[n_id % 3]
            A = np.vstack((source.v[v0], source.v[v1])).T
            tmp_coeffs = np.linalg.lstsq(A, nearest_v, rcond=None)[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest point is a vertex.
            coeffs_v[3 * i + n_id - 4] = 1.0

    matrix = sp.csc_matrix((coeffs_v, (rows, cols)),
                           shape=(target.v.shape[0], source.v.shape[0]))
    return matrix


def qslim_decimator_transformer(mesh, factor=None, n_verts_desired=None):
    """Return simplified faces and a sparse downsampling transform matrix."""
    if factor is None and n_verts_desired is None:
        raise ValueError("Need either factor or n_verts_desired.")

    if n_verts_desired is None:
        n_verts_desired = int(math.ceil(len(mesh.v) * factor))
    n_verts_desired = max(3, int(n_verts_desired))

    Qv = vertex_quadrics(mesh)

    vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
    vert_adj = sp.csc_matrix(
        (np.ones(len(vert_adj)), (vert_adj[:, 0], vert_adj[:, 1])),
        shape=(len(mesh.v), len(mesh.v)))
    vert_adj = (vert_adj + vert_adj.T).tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([[1.0]])))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([[1.0]])))

        destroy_c_cost = float(p1.T.dot(Qsum).dot(p1))
        destroy_r_cost = float(p2.T.dot(Qsum).dot(p2))
        return {
            "destroy_c_cost": destroy_c_cost,
            "destroy_r_cost": destroy_r_cost,
            "collapse_cost": min(destroy_c_cost, destroy_r_cost),
            "Qsum": Qsum,
        }

    queue = []
    for k in range(vert_adj.nnz):
        r = int(vert_adj.row[k])
        c = int(vert_adj.col[k])
        if r > c:
            continue
        cost = collapse_cost(Qv, r, c, mesh.v)["collapse_cost"]
        heapq.heappush(queue, (cost, (r, c)))

    nverts_total = len(mesh.v)
    faces = mesh.f.copy()
    while nverts_total > n_verts_desired and queue:
        e = heapq.heappop(queue)
        r, c = e[1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost["collapse_cost"] > e[0]:
            heapq.heappush(queue, (cost["collapse_cost"], e[1]))
            continue

        if cost["destroy_c_cost"] < cost["destroy_r_cost"]:
            to_destroy = c
            to_keep = r
        else:
            to_destroy = r
            to_keep = c

        np.place(faces, faces == to_destroy, to_keep)

        for idx in range(len(queue)):
            qr, qc = queue[idx][1]
            if qr == to_destroy:
                qr = to_keep
            if qc == to_destroy:
                qc = to_keep
            queue[idx] = (queue[idx][0], (qr, qc))

        Qv[r, :, :] = cost["Qsum"]
        Qv[c, :, :] = cost["Qsum"]

        same_01 = faces[:, 0] == faces[:, 1]
        same_12 = faces[:, 1] == faces[:, 2]
        same_20 = faces[:, 2] == faces[:, 0]
        faces_to_keep = ~(same_01 | same_12 | same_20)
        faces = faces[faces_to_keep, :].copy()

        if faces.size == 0:
            break
        nverts_total = len(np.unique(faces.flatten()))

    if faces.size == 0:
        raise ValueError("Decimation removed all faces; choose a less aggressive factor.")

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    return new_faces, mtx


def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left), num_original_verts))

    return new_faces, mtx


def generate_transform_matrices(mesh, factors):
    """Generate downsampled meshes and transform matrices.

    Returns
    -------
    M : list[SimpleMesh]
    A : list[scipy.sparse.csc_matrix]
    D : list[scipy.sparse.csc_matrix]
    U : list[scipy.sparse.csc_matrix]
    F : list[np.ndarray]
    V : list[np.ndarray]
    """
    if not isinstance(mesh, SimpleMesh):
        mesh = SimpleMesh(v=mesh.x, f=mesh.face)

    factors = [1.0 / x for x in factors]
    M, A, D, U, F, V = [], [], [], [], [], []
    F.append(mesh.f)
    V.append(mesh.v)
    A.append(get_vert_connectivity(mesh.v, mesh.f).astype("float32"))
    M.append(mesh)

    for factor in factors:
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D.astype("float32"))
        new_mesh_v = ds_D.dot(M[-1].v)
        new_mesh = Mesh(v=new_mesh_v, f=ds_f)
        F.append(new_mesh.f)
        V.append(new_mesh.v)
        M.append(new_mesh)
        A.append(get_vert_connectivity(new_mesh.v, new_mesh.f).astype("float32"))
        U.append(setup_deformation_transfer(M[-1], M[-2]).astype("float32"))

    return M, A, D, U, F, V
