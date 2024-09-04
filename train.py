import argparse
import os
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append("..")

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

from utils import load_config, save_codes

class DecayTemperature(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        t = np.exp(-0.00001 * trainer.global_step)
        pl_module.quantizer.temp = t

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default=None)

    args = parser.parse_args()

    config = load_config(args.config_path)

    pl.seed_everything(config.train.seed, workers=True)

    experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_path = os.path.join("./" + config.model.name,
                            config.model.quantizer.name, 
                            config.data.name, experiment_name)
    os.makedirs(log_path, exist_ok=True)

    ckpt_path = os.path.join(log_path, "ckpt")
    os.makedirs(ckpt_path, exist_ok=True)

    code_path = os.path.join(log_path, "codes")
    os.makedirs(code_path, exist_ok=True)

    save_codes(src=".", dst=code_path, cfg=args.config_path)

    logger = TensorBoardLogger(save_dir=log_path, name="logs")

    img_path = os.path.join(log_path, "imgs")
    os.makedirs(img_path, exist_ok=True)

    callbacks = []
    
    callbacks.append(ModelCheckpoint(dirpath=ckpt_path, 
                    save_top_k=3, 
                    monitor='reconst_loss_mse', 
                    mode="min",
                    filename="{epoch:02d}-{step}-{reconst_loss_mse:.5f}"))
    callbacks.append(ModelCheckpoint(dirpath=ckpt_path, 
                    save_top_k=3, 
                    monitor="step", 
                    mode="max",
                    every_n_train_steps=5000,
                    filename="{epoch}-{step}"))
    
    if config.model.name in ["DVAE", "FactorDVAE", "QVAE", "FactorQVAE"]:
        callbacks.extend([DecayTemperature()])

    data = eval(config.data.name)(config)
    model = eval(config.model.name)(config, img_path)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        deterministic=True,
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks,
        max_steps=config.train.max_steps,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch
        )

    trainer.fit(model, data, ckpt_path=args.ckpt_path)
    
if __name__ == "__main__":
    train()