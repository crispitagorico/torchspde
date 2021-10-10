import torch
import torch.nn as nn
import torch.nn.functional as F


###################
# A FNO module which can be used to model a non local operator F
# for F(u(t)) with u(t) a function of 2 space variables
###################

# To use within an SPDEFunc: 
# self._F = F_FNO(modes1, modes2, hidden_channels, hidden_channels, num_layers)


class F_FNO(nn.Module):
    def __init__(self, modes1, modes2, in_channels, hidden_channels, num_layers):
        super(F_FNO, self).__init__()

        """ an FNO to model F(u(t)) where u(t) is a function of 2 spatial variables """

        self.modes1 = modes1
        self.modes2 = modes2

        self.fc0 = nn.Linear(in_channels, hidden_channels)

        self.net = [ FNO_layer(modes1, modes2, hidden_channels) for i in range(num_layers-1) ]
        self.net += [ FNO_layer(modes1, modes2, hidden_channels, last=True) ]
        self.net = nn.Sequential(*self.net)

        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, in_channels)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y, dim_t)"""
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = self.net(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x

###################
# A typical FNO layer
###################

class FNO_layer(nn.Module):
    def __init__(self, modes1, modes2, width, last=False):
        super(FNO_layer, self).__init__()

        self.last = last
        self.conv = ConvolutionSpace(width, modes1, modes2)
        self.w = nn.Linear(width, width)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y, dim_t)"""
        x1 = self.conv(x)
        x2 = self.w(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)           
        return x


###################
# Some helper functions to compute convolutions
###################

def compl_mul2d_spatial(a, b):
    """ ...
    """
    return torch.einsum("aibcd, ijbc -> ajbcd",a,b)

class ConvolutionSpace(nn.Module):
    def __init__(self, channels, modes1, modes2):
        super(ConvolutionSpace, self).__init__()

        """ ...    
        """
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1. / (channels**2)
        self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes1, self.modes2, dtype=torch.cfloat))


    def forward(self, x):
        """ x: (batch, channels, dim_x, dim_y, dim_t)"""

        x0, x1 = x.size(2)//2 - self.modes1//2, x.size(2)//2 + self.modes1//2
        y0, y1 = x.size(3)//2 - self.modes2//2, x.size(3)//2 + self.modes2//2

        # Compute FFT of the input signal to convolve
        x_ft = torch.fft.fftn(x, dim=[2,3])
        x_ft = torch.fft.fftshift(x_ft, dim=[2,3])

        # Pointwise multiplication by complex matrix 
        out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), x.size(4), device=x.device, dtype=torch.cfloat)
        out_ft[:, :, x0:x1, y0:y1, :] = compl_mul2d_spatial(x_ft[:, :, x0:x1, y0:y1], self.weights)

        # Compute Inverse FFT
        out_ft = torch.fft.ifftshift(out_ft, dim=[2,3])
        x = torch.fft.ifftn(out_ft, dim=[2,3], s=(x.size(2), x.size(3)))

        return x.real