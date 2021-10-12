
import torch
import torch.nn as nn
import torch.nn.functional as F

###################
# First some helper objects to compute convolutions
###################

def compl_mul2d(a, b):
    """ ...
    """
    return torch.einsum("aibc, ijbc -> ajbc",a,b)


def compl_mul1d_time(a, b):
    """ ...
    """
    return torch.einsum("aib, ijbc -> ajbc",a,b)

###################
# Now we define the module whose forward pass computes either a space time convolution
# or a spatial convolution with multiple kernels indexed by time
###################

class KernelConvolution(nn.Module):
    def __init__(self, channels, modes1, modes2, T):
        super(KernelConvolution, self).__init__()

        """ This module has a kernel parametrized in the spectral domain
        The method forward computes 
            * a space time convolution if time=True
            * if time=False, the fourier transform inverse of the kernel is computed
              and the input is convolved (in space) with multime kernels indexed by time
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.T = T

        self.scale = 1. / (channels**2)
        self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes1, self.modes2,  dtype=torch.cfloat))
        # self.weights = nn.Parameter(self.scale * torch.rand(channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
    def forward(self, x, time=True):
        """ x: (batch, channels, dim_x, dim_t)"""

        x0, x1 = x.size(2)//2 - self.modes1//2, x.size(2)//2 + self.modes1//2
        t0, t1 = self.T//2 - self.modes2//2, self.T//2 + self.modes2//2

        if time: # If computing the space-time convolution

            # Compute FFT
            x_ft = torch.fft.fftn(x, dim=[2,3])
            x_ft = torch.fft.fftshift(x_ft, dim=[2,3])
 
            # Pointwise multiplication by complex matrix 
            out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), device=x.device, dtype=torch.cfloat)
            out_ft[:, :, x0:x1, t0:t1] = compl_mul2d(x_ft[:, :, x0:x1, t0:t1], self.weights)
            # out_ft[:, :, x0:x1, t0:t1] = x_ft[:, :, x0:x1, t0:t1]*self.weights[None,...]

            # Compute Inverse FFT
            out_ft = torch.fft.ifftshift(out_ft, dim=[2,3])
            x = torch.fft.ifftn(out_ft, dim=[2,3], s=(x.size(2), x.size(3)))
            return x.real

        else: # If computing the convolution in space only
            return self.forward_no_time(x)

    def forward_no_time(self, x):
        """ x: (batch, channels, dim_x)"""

        x0, x1 = x.size(2)//2 - self.modes1//2, x.size(2)//2 + self.modes1//2

        weights = torch.fft.ifftn(torch.fft.ifftshift(self.weights, dim=[-1]), dim=[-1], s=self.T)

        # Compute FFT of the input signal to convolve
        x_ft = torch.fft.fftn(x, dim=[2])
        x_ft = torch.fft.fftshift(x_ft, dim=[2])

        # Pointwise multiplication by complex matrix 
        out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), self.T, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, x0:x1, :] = compl_mul1d_time(x_ft[:, :, x0:x1], weights)
        # out_ft[:, :, x0:x1, :] = x_ft[:, :, x0:x1][...,None]*weights[None,...]

        # Compute Inverse FFT
        out_ft = torch.fft.ifftshift(out_ft, dim=[2])
        x = torch.fft.ifftn(out_ft, dim=[2], s=x.size(2))

        return x.real


###################
# Now we define the SPDEs solver
#
# We begin by defining the fixed point map.
###################

class IterationLayer(nn.Module):
    def __init__(self, spde_func, modes1, modes2, T):
        super(IterationLayer, self).__init__()

        self._spde = spde_func
        self.convolution = KernelConvolution(spde_func._hidden_size, modes1, modes2, T)  

    def forward(self, u, xi):
        """ - u: (batch, hidden_channels, dim_x, dim_t)
            - xi: (batch, forcing_channels, dim_x, dim_t)
        """
        Fu, Gu = self._spde(u) 
        Guxi = torch.einsum('abcde, acde -> abde', Gu, xi)
        sol = Fu + Guxi
        return self.convolution(sol)

###################
# Now we wrap it up into something that solves the SPDE.
###################

class NeuralFixedPoint(nn.Module):
    def __init__(self, spde_func, modes1, modes2, T, n_iter):
        super(NeuralFixedPoint, self).__init__()

        # self.padding = int(2**(np.ceil(np.log2(abs(2*T-1)))))

        self.n_iter = n_iter
        
        self.iter_layer = IterationLayer(spde_func, modes1, modes2, T) 
      
        self.initial_convolution = self.iter_layer.convolution

    def forward(self, z0, xi):
        """ - z0: (batch, hidden_channels, dim_x)
            - xi: (batch, forcing_channels, dim_x, dim_t)
        """
        
        # x = F.pad(x,[0,self.padding-10])
        z0 =  self.initial_convolution(z0, time=False) 

        z = z0
        for i in range(self.n_iter):
            y = z0 + self.iter_layer(z, xi)
            z = y
        
        return y