import argparse
import os
import glob
import numpy as np

import torch
import torchvision

from data.shapes3d import Shapes3DDataModule
from data.isaac3d import Isaac3DDataModule
from data.mpi3d_complex import MPI3DComplexDataModule

from models.ae import AE
from models.bioae import BioAE
from models.dvae import DVAE
from models.factor_dvae import FactorDVAE
from models.factor_qvae import FactorQVAE
from models.factor_vae import FactorVAE
from models.qlae import QLAE
from models.qvae import QVAE
from models.vae import VAE

from utils import load_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Latent Traversal")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantizer", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    random_state = np.random.RandomState(1)
    args = parser.parse_args()

    exp_path = os.path.join(args.model, args.quantizer, 
                            args.data, args.exp_name)
    config_path = glob.glob(os.path.join(exp_path, "codes", "*.yaml"))[0]
    ckpt_path = os.path.join(exp_path, "ckpt", args.ckpt)

    config = load_config(config_path)

    data_module = eval(config.data.name)(config)
    data_module.setup()
    dataset = data_module.dataset

    model = eval(config.model.name)(config, img_dir=None).cuda()
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    num_latents = config.data.num_latents

    factors = dict()

    for factor_id in range(dataset.num_total_factors):
        factors[factor_id] = random_state.randint(0, dataset.factor_sizes[factor_id], 5)
    
    print(factors)
    img_values = list(zip(*factors.values()))
    
    for img_value in img_values:
        img_index = int(np.dot(np.array(img_value), dataset.factor_bases))
        reconst_path = os.path.join(exp_path, 'latent_traversal', str(img_value))
        os.makedirs(reconst_path, exist_ok=True)

        img = dataset.__getitem__(img_index)
        img = img.unsqueeze(0).cuda()

        if model.config.model.name in ["AE", "BioAE"]:
            z = model.encoder(img)
        elif model.config.model.name in ["VAE", "FactorVAE"]:
            logits = model.encoder(img)
            mu = logits[:, :model.config.data.num_latents]
            logvar = logits[:, model.config.data.num_latents:]
            z = model.reparameterize(mu, logvar)
        elif model.config.model.name == "QLAE":
            logits = model.encoder(img)
            z, _, _ = model.quantizer(logits, 0)
        elif model.config.model.name in ["QVAE", "FactorQVAE", "DVAE", "FactorDVAE"]:
            z_e = model.encoder(img)
            z, _, _, _ = model.quantizer(z_e, 0)
        z = z.detach().cpu().numpy().reshape(-1)

        for d in range(num_latents):
            dim_path = os.path.join(reconst_path, 'dim_{}'.format(d))
            os.makedirs(dim_path, exist_ok=True)
            torchvision.utils.save_image(img, os.path.join(dim_path, "{}.png".format(img_index)), normalize=True, padding=0)  
            if model.config.model.name in ["AE", "BioAE", "VAE", "FactorVAE"]:
                traverse = np.linspace(z[d] - 3, z[d] + 3, 8)  
            elif model.config.model.name == "QLAE":
                codebook = model.quantizer.codebook[d].data
                min_val = codebook.min().detach().cpu().numpy()
                max_val = codebook.max().detach().cpu().numpy()                         
                traverse = np.linspace(min_val, max_val, 8)
            elif model.config.model.name in ["QVAE", "FactorQVAE", "DVAE", "FactorDVAE"]:
                codebook = model.quantizer.codebook.weight.data
                min_val = codebook.min().detach().cpu().numpy()
                max_val = codebook.max().detach().cpu().numpy()                         
                traverse = np.linspace(min_val, max_val, 8)
            for i in traverse:
                if d == 0:
                    z_traversed = (torch.from_numpy(np.concatenate((np.array([i]), z[1:])))).unsqueeze(0).cuda()
                elif d == num_latents - 1:
                    z_traversed = (torch.from_numpy(np.concatenate((z[:-1], np.array([i]))))).unsqueeze(0).cuda()
                else:
                    z_traversed = (torch.from_numpy(np.concatenate((np.concatenate((z[:d], np.array([i]))), np.array(z[d+1:]))))).unsqueeze(0).cuda()
    
                z_traversed = z_traversed.squeeze(0).float()
                # print(z_traversed.shape)
                reconst = model.decoder(z_traversed)
                torchvision.utils.save_image(reconst, os.path.join(dim_path, "{}_{}.png".format(img_index, i)), normalize=True, padding=0)