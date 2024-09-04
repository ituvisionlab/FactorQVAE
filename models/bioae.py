import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

from encoders.bioae_encoder import BioAEEncoder
from decoders.bioae_decoder import BioAEDecoder

from models.base import Base

class BioAE(Base):

    def __init__(self, config, img_dir):
        super(BioAE, self).__init__(config, img_dir)

    def nonnegative_loss(self, logits):
        return torch.mean(torch.maximum(-logits, torch.tensor(0.0)))
    
    def activity_loss(self, logits):
        return torch.mean(torch.square(logits))

    def forward(self, img, iter):
        """
        Encode image, then reconstruct image
        """
        logits = self.encoder(img)
        nonneg_loss = self.nonnegative_loss(logits)
        activity_loss = self.activity_loss(logits)
        reconst_img = self.decoder(logits)

        return reconst_img, nonneg_loss, activity_loss
        
    def training_step(self, img, batch_idx):
        iter = self.global_step
        
        reconst_img, nonneg_loss, activity_loss = self.forward(img, iter)        
        # mean squared loss for reconstruction
        reconst_loss = eval(self.config.train.reconst_loss)(reconst_img, img)
        
        loss = reconst_loss + self.config.model.quantizer.nonneg_coeff * nonneg_loss \
                            + self.config.model.quantizer.activity_coeff * activity_loss

        self.log('reconst_loss_mse', reconst_loss, prog_bar=True)
        self.log('nonneg_loss', self.config.model.quantizer.nonneg_coeff * nonneg_loss, prog_bar=True)
        self.log('activity_loss', self.config.model.quantizer.activity_coeff * activity_loss, prog_bar=True)
        self.log('loss', loss, prog_bar=True)

        if iter % 500 == 0:
            torchvision.utils.save_image(img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_real.png"), normalize=True, padding=0)
            torchvision.utils.save_image(reconst_img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_reconst.png"), normalize=True, padding=0)

        return loss
    
    def validation_step(self, img, batch_idx):
        iter = self.global_step
        
        reconst_img, nonneg_loss, activity_loss = self.forward(img, iter)        
        # mean squared loss for reconstruction
        reconst_loss = eval(self.config.train.reconst_loss)(reconst_img, img)
        
        loss = reconst_loss + self.config.model.quantizer.nonneg_coeff * nonneg_loss \
                            + self.config.model.quantizer.activity_coeff * activity_loss

        self.log('test_reconst_loss_mse', reconst_loss, prog_bar=True)
        self.log('test_nonneg_loss', self.config.model.quantizer.nonneg_coeff * nonneg_loss, prog_bar=True)
        self.log('test_activity_loss', self.config.model.quantizer.activity_coeff * activity_loss, prog_bar=True)
        self.log('test_loss', loss, prog_bar=True)

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