import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelConvolution(nn.Module):
    def __init__(self, channels, modes1, T):
        super(KernelConvolution, self).__init__()

        """ ...    
        """
        self.modes1 = modes1
        self.T = T

        self.scale = 1. / (channels**2)
        self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes1, dtype=torch.cfloat))
        # self.weights = nn.Parameter(self.scale * torch.rand(channels, self.modes1, dtype=torch.cfloat))
        
    def forward(self, x, time=True):
        """ x: (batch, channels, dim_t)"""

        t0, t1 = self.T//2 - self.modes3//2, self.T//2 + self.modes3//2

        # Compute FFT
        x_ft = torch.fft.fftn(x, dim=[2])
        x_ft = torch.fft.fftshift(x_ft, dim=[2])
 
        # Pointwise multiplication by complex matrix 
        out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), device=x.device, dtype=torch.cfloat)
        out_ft[:, :, t0:t1] = torch.einsum("aib, ijb -> ajb", x_ft[:, :, t0:t1], self.weights) 
        # out_ft[:, :, t0:t1] = x_ft[:, :, t0:t1]*self.weights[None,...]

        # Compute Inverse FFT
        out_ft = torch.fft.ifftshift(out_ft, dim=[2])
        x = torch.fft.ifftn(out_ft, dim=[2], s=(x.size(2)))
        return x.real


class F(nn.Module):
    def __init__(self, hidden_channels, forcing_channels):
        super(F, self).__init__()
        """ ...
        """
        self.forcing_channels = forcing_channels

        # net = [nn.Linear(hidden_channels, hidden_channels*forcing_channels), nn.Tanh()]
        net = [nn.Conv1d(hidden_channels, hidden_channels*forcing_channels, 1), nn.BatchNorm1d(hidden_channels*forcing_channels), nn.Tanh()] 
        self.net = nn.Sequential(*net)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_t)"""
        return self.net(x).view(x.size(0), x.size(1), self.forcing_channels, x.size(2))


class IterationLayer(nn.Module):
    def __init__(self, modes1, hidden_channels, forcing_channels, T):
        super(IterationLayer, self).__init__()
        """...
        """

        self.F = F(hidden_channels, forcing_channels)
        self.convolution = KernelConvolution(hidden_channels, modes1, T)

    def forward(self, x, xi):
        """ - x: (batch, hidden_channels, dim_t)
            - xi: (batch, forcing_channels, dim_t)
        """
        mat = self.F(x)
        y = torch.einsum('abcd, acd -> abd', mat, xi)
        return self.convolution(y)
        

class NeuralFixedPoint(nn.Module):
    def __init__(self, modes1, in_channels, hidden_channels, forcing_channels, out_channels, T, n_iter):
        super(NeuralFixedPoint, self).__init__()

        """ ...
        """

        # self.padding = int(2**(np.ceil(np.log2(abs(2*T-1)))))

        self.n_iter = n_iter
        
        self.readin = nn.Linear(in_channels, hidden_channels)
        
        self.iter_layer = IterationLayer(modes1, hidden_channels, forcing_channels, T) 
      
        readout = [nn.Linear(hidden_channels, 128), nn.ReLU(), nn.Linear(128, out_channels)]
        self.readout = nn.Sequential(*readout)

    def forward(self, x, xi):
        """ - x: (batch, in_channels)
            - xi: (batch, forcing_channels, dim_t)
        """

        z0 = self.readin(x)

        z = z0
        for i in range(self.n_iter):
            y = z0 + self.iter_layer(z, xi)
            z = y
        
        return self.readout(y.permute(0,2,1)).permute(0,2,1)