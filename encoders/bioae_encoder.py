from encoders.ae_encoder import AEEncoder

class BioAEEncoder(AEEncoder):
    def __init__(self, config):
        super(BioAEEncoder, self).__init__(config)