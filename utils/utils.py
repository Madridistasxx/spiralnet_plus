import os

import numpy as np
import torch
from torch_geometric.data import Data


def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_sparse(spmat):
    coo = spmat.tocoo()
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64)
    )
    values = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def _face_to_f3(face):
    face_np = face.detach().cpu().numpy() if hasattr(face, "detach") else np.asarray(face)
    face_np = face_np.astype(np.int64, copy=False)
    if face_np.ndim != 2:
        raise ValueError(f"face must be 2D, got shape {face_np.shape}")
    # Accept both common formats: [F, 3] and PyG [3, F].
    if face_np.shape[0] == 3 and face_np.shape[1] != 3:
        face_np = face_np.T
    if face_np.shape[1] != 3:
        raise ValueError(f"face must have shape [F, 3] or [3, F], got {face_np.shape}")
    return face_np


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    """Generate spiral indices without OpenMesh.

    Args:
        face: triangular faces in either [F, 3] or PyG [3, F] format.
        seq_length: spiral length.
        vertices: optional vertex coordinates [N, 3]. If omitted, dummy
            coordinates are used; topology still comes from face.
        dilation: spiral dilation.
    """
    from .generate_spiral_seq_no_om import extract_spirals_from_arrays

    face_np = _face_to_f3(face)

    if vertices is not None:
        vertices_np = vertices.detach().cpu().numpy() if hasattr(vertices, "detach") else np.asarray(vertices)
        vertices_np = vertices_np.astype(np.float64, copy=False)
    else:
        n_vertices = int(face_np.max()) + 1
        vertices_np = np.ones((n_vertices, 3), dtype=np.float64)

    spirals = extract_spirals_from_arrays(
        vertices=vertices_np,
        faces=face_np,
        edge_index=None,
        seq_length=seq_length,
        dilation=dilation,
    )
    return torch.tensor(spirals, dtype=torch.long)


def pyg_data_to_obj(data, out_path="template.obj"):
    """
    Convert PyG Data(x, face) to Wavefront OBJ.

    data.x:    [N, 3]
    data.face: [3, F] or [F, 3]
    """
    x = data.x.detach().cpu()
    face = _face_to_f3(data.face)

    with open(out_path, "w") as f:
        f.write("# template.obj generated from PyG Data\n")

        for v in x:
            f.write(f"v {v[0].item():.8f} {v[1].item():.8f} {v[2].item():.8f}\n")

        # OBJ indices start from 1; PyTorch/PyG indices start from 0.
        for tri in face:
            i, j, k = tri.tolist()
            f.write(f"f {i + 1} {j + 1} {k + 1}\n")

    print(f"Saved to {out_path}")


def obj_to_pyg_data(obj_path, build_edge_index=False):
    vertices = []
    faces = []

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = [int(p.split("/")[0]) - 1 for p in parts]
                if len(face) == 3:
                    faces.append(face)
                elif len(face) == 4:
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[0], face[2], face[3]])
                else:
                    raise ValueError(f"Only triangular/quad faces are supported, got {len(face)} vertices")

    x = torch.tensor(vertices, dtype=torch.float32)
    face = torch.tensor(faces, dtype=torch.long).contiguous()
    data = Data(x=x, face=face)

    if build_edge_index:
        data.edge_index = build_edges_from_faces(face)

    return data


def build_edges_from_faces(face):
    face_np = _face_to_f3(face)
    edges = set()

    for i, j, k in face_np.tolist():
        edges.add((i, j)); edges.add((j, i))
        edges.add((j, k)); edges.add((k, j))
        edges.add((k, i)); edges.add((i, k))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
