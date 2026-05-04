import utils
import torch
import numpy as np

dies = torch.load("/mnt/data/datasets/ddacs/graph/dies.pt", weights_only=False)

origin_die0 = dies[0]
obj_path = "/home/RUS_CIP/st189459/spiralnet_plus/data/ddacs/template/template_die.obj"
recon_die0 = utils.obj_to_pyg_data(obj_path)

print(origin_die0)
print(recon_die0)
