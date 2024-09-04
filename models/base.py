import torch.nn as nn

import pytorch_lightning as pl

from encoders.qlae_encoder import QLAEEncoder
from decoders.qlae_decoder import QLAEDecoder

from encoders.dvae_encoder import DVAEEncoder
from decoders.dvae_decoder import DVAEDecoder

from encoders.factor_dvae_encoder import FactorDVAEEncoder
from decoders.factor_dvae_decoder import FactorDVAEDecoder

from encoders.ae_encoder import AEEncoder
from decoders.ae_decoder import AEDecoder

from encoders.vae_encoder import VAEEncoder
from decoders.vae_decoder import VAEDecoder

from encoders.qvae_encoder import QVAEEncoder
from decoders.qvae_decoder import QVAEDecoder

from encoders.factor_qvae_encoder import FactorQVAEEncoder
from decoders.factor_qvae_decoder import FactorQVAEDecoder

from encoders.factor_vae_encoder import FactorVAEEncoder
from decoders.factor_vae_decoder import FactorVAEDecoder

from encoders.bioae_encoder import BioAEEncoder
from decoders.bioae_decoder import BioAEDecoder

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1 or classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Base(pl.LightningModule):

    def __init__(self, config, img_dir=None):
        super().__init__()

        self.config = config
        self.img_dir = img_dir
        
        encoder_name = self.config.model.name + "Encoder"
        self.encoder = eval(encoder_name)(self.config)

        decoder_name = self.config.model.name + "Decoder"
        self.decoder = eval(decoder_name)(self.config)

        self.apply(weights_init)
    
    def forward(self, img, iter):
        raise NotImplementedError()   
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError()
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()
    
    def configure_optimizers(self):
        raise NotImplementedError()