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
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    from .generate_spiral_seq import extract_spirals
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals


def pyg_data_to_obj(data, out_path="template.obj"):
    """
    Convert PyG Data(x, face) to Wavefront OBJ.
    
    data.x:    [N, 3]
    data.face: [3, F] or [F, 3]
    """
    x = data.x.detach().cpu()
    face = data.face.detach().cpu().T

    if face.shape[1] == 4:
        face = np.concatenate([face[:, [0, 1, 2]], face[:, [0, 2, 3]]])

    if face.shape[0] == 3:
        face = face.t()   # [F, 3]

    with open(out_path, "w") as f:
        f.write("# template.obj generated from PyG Data\n")

        # vertices
        for v in x:
            f.write(f"v {v[0].item():.8f} {v[1].item():.8f} {v[2].item():.8f}\n")

        # faces
        # OBJ index start from 1，PyTorch/PyG from 0
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
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])

            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face)

    x = torch.tensor(vertices, dtype=torch.float32)

    face = torch.tensor(faces, dtype=torch.long).contiguous()

    data = Data(x=x, face=face)

    if build_edge_index:
        edge_index = build_edges_from_faces(face)
        data.edge_index = edge_index

    return data

def build_edges_from_faces(face):
    edges = set()

    for tri in face:
        i, j, k = tri.tolist()

        edges.add((i, j)); edges.add((j, i))
        edges.add((j, k)); edges.add((k, j))
        edges.add((k, i)); edges.add((i, k))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t()
    return edge_index