import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

import pytorch_lightning as pl

from encoders.factor_vae_encoder import FactorVAEEncoder
from decoders.factor_vae_decoder import FactorVAEDecoder
from blocks.discriminator import Discriminator

from models.base import Base, weights_init

def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

class FactorVAE(Base):

    def __init__(self, config, img_dir):
        super(FactorVAE, self).__init__(config, img_dir)
        self.kl_weight = self.config.model.quantizer.kl_weight
        self.gamma = self.config.model.discriminator.gamma
        self.automatic_optimization = False
        self.discriminator = Discriminator(latent_dim=self.config.data.num_latents)
        self.discriminator.apply(weights_init)

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

        return reconst_img, z, mu, logvar
        
    def training_step(self, img, batch_idx):
        vae_optimizer, discriminator_optimizer = self.optimizers()

        # Use first half of the batch for VAE update, second half of the batch for Discriminator update
        batch_size = img.shape[0] // 2

        first_batch = img[:batch_size,:,:,:]
        second_batch = img[batch_size:,:,:,:]        
        iter = self.global_step
        
        ################
        # Optimize VAE #
        ################
        reconst_img, z, mu, logvar = self.forward(first_batch, iter)        
        # mean squared loss for reconstruction
        reconst_loss = eval(self.config.train.reconst_loss)(reconst_img, first_batch)
        # KL divergence loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        
        D_z = self.discriminator(z)
        vae_tc_loss = (D_z[:, 0] - D_z[:, 1]).mean()
        
        vae_loss = reconst_loss + self.kl_weight * kl_loss + self.gamma * vae_tc_loss

        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)
        vae_optimizer.step()

        ##########################
        # Optimize Discriminator #
        ##########################
        ones = torch.ones(batch_size, dtype=torch.long, requires_grad=False, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, requires_grad=False, device=self.device)

        _, second_batch_z, _, _ = self.forward(second_batch, iter)
        permuted_z = permute_dims(second_batch_z).detach()
        D_perm = self.discriminator(permuted_z)

        disc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_perm, ones))

        discriminator_optimizer.zero_grad()
        self.manual_backward(disc_loss)
        discriminator_optimizer.step()

        vae_lr_scheduler, discriminator_lr_scheduler = self.lr_schedulers()
        vae_lr_scheduler.step()
        discriminator_lr_scheduler.step()

        self.log('reconst_loss_mse', reconst_loss, prog_bar=True)
        self.log('additional_loss', self.kl_weight * kl_loss, prog_bar=True)
        self.log('only_kl', kl_loss, prog_bar=True)
        self.log('vae_tc_loss', self.gamma *vae_tc_loss, prog_bar=True)
        self.log('only_vae_tc_loss', vae_tc_loss, prog_bar=True)
        self.log('vae_loss', vae_loss, prog_bar=True)
        self.log('disc_loss', disc_loss, prog_bar=True)

        if iter % 500 == 0:
            torchvision.utils.save_image(img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_real.png"), normalize=True, padding=0)
            torchvision.utils.save_image(reconst_img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_reconst.png"), normalize=True, padding=0)
    
    def validation_step(self, img, batch_idx):
        batch_size = img.shape[0] // 2

        first_batch = img[:batch_size,:,:,:]
        second_batch = img[batch_size:,:,:,:]        
        iter = self.global_step
        
        reconst_img, z, mu, logvar = self.forward(first_batch, iter)        
        # mean squared loss for reconstruction
        reconst_loss = eval(self.config.train.reconst_loss)(reconst_img, first_batch)
        # KL divergence loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        
        D_z = self.discriminator(z)
        vae_tc_loss = (D_z[:, 0] - D_z[:, 1]).mean()
        
        vae_loss = reconst_loss + self.kl_weight * kl_loss + self.gamma * vae_tc_loss

        ones = torch.ones(batch_size, dtype=torch.long, requires_grad=False, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, requires_grad=False, device=self.device)

        _, second_batch_z, _, _ = self.forward(second_batch, iter)
        permuted_z = permute_dims(second_batch_z).detach()
        D_perm = self.discriminator(permuted_z)

        disc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_perm, ones))

        self.log('test_reconst_loss_mse', reconst_loss, prog_bar=True)
        self.log('test_additional_loss', self.kl_weight * kl_loss, prog_bar=True)
        self.log('test_only_kl', kl_loss, prog_bar=True)
        self.log('test_vae_tc_loss', self.gamma *vae_tc_loss, prog_bar=True)
        self.log('test_only_vae_tc_loss', vae_tc_loss, prog_bar=True)
        self.log('test_vae_loss', vae_loss, prog_bar=True)
        self.log('test_disc_loss', disc_loss, prog_bar=True)

        torchvision.utils.save_image(img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_test_real.png"), normalize=True, padding=0)
        torchvision.utils.save_image(reconst_img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_test_reconst.png"), normalize=True, padding=0)
    
    def configure_optimizers(self):
        vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        vae_optimizer = eval(self.config.train.optimizer)(vae_params, lr=self.config.train.vae_lr)
        discriminator_optimizer = eval(self.config.train.optimizer)(self.discriminator.parameters(), lr=self.config.train.disc_lr)

        return (
            {
                "optimizer": vae_optimizer,
                "lr_scheduler": {
                    "scheduler": CosineAnnealingLR(vae_optimizer, T_max=self.config.train.vae_lr_iter, eta_min=1.25e-6),
                    "interval": "step",
                    "frequency": 1
                },
            },
            {
                "optimizer": discriminator_optimizer,
                "lr_scheduler": {
                    "scheduler": CosineAnnealingLR(discriminator_optimizer, T_max=self.config.train.disc_lr_iter, eta_min=1.25e-6),
                    "interval": "step",
                    "frequency": 1
                },
            },
        )