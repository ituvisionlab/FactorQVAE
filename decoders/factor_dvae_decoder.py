from decoders.dvae_decoder import DVAEDecoder

class FactorDVAEDecoder(DVAEDecoder):
    def __init__(self, config):
        super(FactorDVAEDecoder, self).__init__(config)