import argparse
import os
import glob
import numpy as np

import torch

import plotly.graph_objects as go
import pickle as pkl

from data.shapes3d import Shapes3DDataModule
from data.isaac3d import Isaac3DDataModule
from data.mpi3d_complex import MPI3DComplexDataModule

from models.dvae import DVAE
from models.factor_dvae import FactorDVAE
from models.qvae import QVAE
from models.factor_qvae import FactorQVAE

from utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantizer", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    
    args = parser.parse_args()

    exp_path = os.path.join(args.model, args.quantizer, 
                            args.data, args.exp_name)
    config_path = glob.glob(os.path.join(exp_path, "codes", "*.yaml"))[0]
    ckpt_path = os.path.join(exp_path, "ckpt", args.ckpt)

    config = load_config(config_path)

    data_module = eval(config.data.name)(config)
    data_module.setup()
    data_loader = data_module.test_dataloader()

    model = eval(config.model.name)(config, img_dir=None).cuda()
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    hist_for_each_position_path = os.path.join(exp_path, 'hist_for_each_position')
    os.makedirs(hist_for_each_position_path, exist_ok=True)

    hist_for_each_position = dict()

    num_embeddings = config.model.codebook.num_embeddings
    latent_positions_num = config.data.num_latents

    for i in range(latent_positions_num):
        hist_for_each_position.update({i: np.zeros(num_embeddings)})

    for i, batch in enumerate(data_loader):
        batch = batch.cuda()
        z_e = model.encoder(batch)
        _, _, indices, _ = model.quantizer(z_e, 0)
        indices = indices.argmax(dim=-1)  
        
        indices = indices.detach().cpu().numpy()
        
        for single_img_indices in indices:
            for pos, index in enumerate(single_img_indices):
                hist_for_each_position[pos][index] += 1
    with open(os.path.join(exp_path, "hist_for_each_position.pkl"), "wb") as f:
        pkl.dump(hist_for_each_position, f)
    
    for idx, hist in enumerate(hist_for_each_position.values()):
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                texttemplate="%{x} : %{y}",
                textposition="outside",
                x=[i for i in range(num_embeddings)],
                y=hist,
                width=1
            )
        )
        fig.update_layout(
            autosize=True
        )
        fig.write_html(hist_for_each_position_path + "/" + str(idx) + ".html")