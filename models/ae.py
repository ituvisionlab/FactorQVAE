import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

import pytorch_lightning as pl

from encoders.ae_encoder import AEEncoder
from decoders.ae_decoder import AEDecoder

from models.base import Base

class AE(Base):

    def __init__(self, config, img_dir):
        super(AE, self).__init__(config, img_dir)

    def forward(self, img, iter):
        """
        Encode image, then reconstruct image
        """
        logits = self.encoder(img)
        reconst_img = self.decoder(logits)

        return reconst_img 
        
    def training_step(self, img, batch_idx):
        iter = self.global_step
        
        reconst_img = self.forward(img, iter)        
        # mean squared loss for reconstruction
        loss = eval(self.config.train.reconst_loss)(reconst_img, img)

        self.log('reconst_loss_mse', loss, prog_bar=True)

        if iter % 500 == 0:
            torchvision.utils.save_image(img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_real.png"), normalize=True, padding=0)
            torchvision.utils.save_image(reconst_img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_reconst.png"), normalize=True, padding=0)

        return loss
    
    def validation_step(self, img, batch_idx):
        iter = self.global_step
        
        reconst_img = self.forward(img, iter)
        
        loss = eval(self.config.train.reconst_loss)(reconst_img, img)

        self.log('test_loss', loss, prog_bar=False)

        torchvision.utils.save_image(img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_test_real.png"), normalize=True, padding=0)
        torchvision.utils.save_image(reconst_img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_test_reconst.png"), normalize=True, padding=0)
    
    def configure_optimizers(self):
        optimizer = eval(self.config.train.optimizer)(self.parameters(), lr=self.config.train.vae_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(optimizer, T_max=self.config.train.vae_lr_iter, eta_min=1.25e-6),
                "interval": "step",
                "frequency": 1
            },
        }