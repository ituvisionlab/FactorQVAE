from torch import nn

class DenseBlock(nn.Module):
    """
    A dense block of two linear layers with ReLU activation.

    Args:
        in_channel (int): The number of input channels.
        out_channel (int): The number of output channels.
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()

        modules = []
        
        modules.append(nn.Linear(in_channel, out_channel))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(out_channel, out_channel))
        modules.append(nn.ReLU(inplace=True))             

        self.dense_block = nn.Sequential(*modules)
    
    def forward(self, x):
        out = self.dense_block(x)
        
        return out