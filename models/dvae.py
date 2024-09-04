import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

from encoders.dvae_encoder import DVAEEncoder
from decoders.dvae_decoder import DVAEDecoder
from quantizers.gumbel_quantizer import GumbelQuantizer

from models.base import Base

class DVAE(Base):

    def __init__(self, config, img_dir):
        super(DVAE, self).__init__(config, img_dir)
        
        self.quantizer = eval(self.config.model.quantizer.name)(config=self.config)
        self.kl_weight = self.config.model.quantizer.kl_weight

    def forward(self, img, iter):
        """
        Encode image, quantize tensor, reconstruct image
        iter is the current iteration of the training;
        necessary for kl weight annealing
        """
        logits = self.encoder(img)
        # aux_loss is commitment loss for VQVAE like archs, kl loss for dVAE like archs
        z_q, aux_loss, indices, perplexity = self.quantizer(logits, iter)
        reconst_img = self.decoder(z_q)

        return reconst_img, aux_loss, indices, perplexity   
        
    def training_step(self, img, batch_idx):
        iter = self.global_step
        
        reconst_img, aux_loss, indices, perplexity = self.forward(img, iter)        
        # mean squared loss for reconstruction
        reconst_loss = eval(self.config.train.reconst_loss)(reconst_img, img)
        
        loss = reconst_loss + self.kl_weight * aux_loss

        self.log('reconst_loss_mse', reconst_loss, prog_bar=True)
        self.log('additional_loss', self.kl_weight * aux_loss, prog_bar=True)
        self.log('loss', loss, prog_bar=True)
        self.log('perplexity', perplexity, prog_bar=True)

        if iter % 500 == 0:
            torchvision.utils.save_image(img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_real.png"), normalize=True, padding=0)
            torchvision.utils.save_image(reconst_img[0,:,:,:], os.path.join(self.img_dir, str(iter)+"_reconst.png"), normalize=True, padding=0)

        return loss
    
    def validation_step(self, img, batch_idx):
        iter = self.global_step
        
        reconst_img, aux_loss, indices, perplexity = self.forward(img, iter)
        
        reconst_loss = eval(self.config.train.reconst_loss)(reconst_img, img)
        loss = reconst_loss + self.kl_weight * aux_loss

        self.log('test_reconst_loss', reconst_loss, prog_bar=True)
        self.log('test_additional_loss', self.kl_weight * aux_loss, prog_bar=False)
        self.log('test_loss', loss, prog_bar=False)
        self.log('test_perplexity', perplexity, prog_bar=True)

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