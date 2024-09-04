import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

from encoders.vae_encoder import VAEEncoder
from decoders.vae_decoder import VAEDecoder

from models.base import Base

class VAE(Base):

    def __init__(self, config, img_dir):
        super(VAE, self).__init__(config, img_dir)
        self.kl_weight = self.config.model.quantizer.kl_weight

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample z values
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, img, iter):
        """
        Encode image, then reconstruct image
        """
        logits = self.encoder(img)
        mu = logits[:, :self.config.data.num_latents]
        logvar = logits[:, self.config.data.num_latents:]
        z = self.reparameterize(mu, logvar)
        reconst_img = self.decoder(z)

        return reconst_img, mu, logvar
        
    def training_step(self, img, batch_idx):
        iter = self.global_step
        
        reconst_img, mu, logvar = self.forward(img, iter)        
        # mean squared loss for reconstruction
        reconst_loss = eval(self.config.train.reconst_loss)(reconst_img, img)
        # KL divergence loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        
        loss = reconst_loss + self.kl_weight * kl_loss

        self.log('reconst_loss_mse', reconst_loss, prog_bar=True)
        self.log('additional_loss', self.kl_weight * kl_loss, prog_bar=True)
        self.log('only_kl', kl_loss, prog_bar=True)
        self.log('loss', loss, prog_bar=True)

        if iter % 500 == 0:
            torchvision.utils.save_image(img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_real.png"), normalize=True, padding=0)
            torchvision.utils.save_image(reconst_img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_reconst.png"), normalize=True, padding=0)

        return loss
    
    def validation_step(self, img, batch_idx):
        iter = self.global_step
        
        reconst_img, mu, logvar = self.forward(img, iter)
        reconst_loss = eval(self.config.train.reconst_loss)(reconst_img, img)

        kl_loss = self.kl_weight * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        
        loss = reconst_loss + kl_loss

        self.log('test_reconst_loss', reconst_loss, prog_bar=True)
        self.log('test_additional_loss', kl_loss, prog_bar=True)
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