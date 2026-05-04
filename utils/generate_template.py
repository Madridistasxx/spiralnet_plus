import torch
import utils
from pathlib import Path

def create_template():
    print("Loading data...")
    dies = torch.load("/mnt/data/datasets/ddacs/graph/dies.pt", weights_only=False)
    punches = torch.load("/mnt/data/datasets/ddacs/graph/punches.pt", weights_only=False)
    binders = torch.load("/mnt/data/datasets/ddacs/graph/binders.pt", weights_only=False)

    die0 = dies[0]
    punch0 = punches[0]
    binder0 = binders[0]

    root_dir = (Path(__file__).parent.parent)
    data_dir = root_dir / "data" / "ddacs" / "template"
    utils.makedirs(data_dir)

    fp_die = data_dir / "template_die.obj"
    fp_punch = data_dir / "template_punch.obj"
    fp_binder = data_dir / "template_binder.obj"

    print("Creating template.obj...")
    utils.pyg_data_to_obj(die0, fp_die)
    utils.pyg_data_to_obj(punch0, fp_punch)
    utils.pyg_data_to_obj(binder0, fp_binder)

