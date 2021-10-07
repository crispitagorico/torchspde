
import torch
import torch.nn as nn
import torch.nn.functional as F

###################
# First some helper objects to compute convolutions
###################

def compl_mul3d(a, b):
    """ ...
    """
    return torch.einsum("aibcd, ijbcd -> ajbcd",a,b)


def compl_mul2d_time(a, b):
    """ ...
    """
    return torch.einsum("aibc, ijbcd -> ajbcd",a,b)

def compl_mul2d_spatial(a, b):
    """ ...
    """
    return torch.einsum("aibcd, ijbc -> ajbcd",a,b)


class KernelConvolution(nn.Module):
    def __init__(self, channels, modes1, modes2, modes3, T):
        super(KernelConvolution, self).__init__()

        """ ...    
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.T = T

        self.scale = 1. / (channels**2)
        self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        # self.weights = nn.Parameter(self.scale * torch.rand(channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        
    def forward(self, x, time=True):
        """ x: (batch, channels, dim_x, dim_y, dim_t)"""

        x0, x1 = x.size(2)//2 - self.modes1//2, x.size(2)//2 + self.modes1//2
        y0, y1 = x.size(3)//2 - self.modes2//2, x.size(3)//2 + self.modes2//2
        t0, t1 = self.T//2 - self.modes3//2, self.T//2 + self.modes3//2

        if time: # If computing the space-time convolution

            # Compute FFT
            x_ft = torch.fft.fftn(x, dim=[2,3,4])
            x_ft = torch.fft.fftshift(x_ft, dim=[2,3,4])
 
            # Pointwise multiplication by complex matrix 
            out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), x.size(4), device=x.device, dtype=torch.cfloat)
            out_ft[:, :, x0:x1, y0:y1, t0:t1] = compl_mul3d(x_ft[:, :, x0:x1, y0:y1, t0:t1], self.weights)
            # out_ft[:, :, x0:x1, y0:y1, t0:t1] = x_ft[:, :, x0:x1, y0:y1, t0:t1]*self.weights[None,...]

            # Compute Inverse FFT
            out_ft = torch.fft.ifftshift(out_ft, dim=[2,3,4])
            x = torch.fft.ifftn(out_ft, dim=[2,3,4], s=(x.size(-3), x.size(-2), x.size(-1)))
            return x.real

        else: # If computing the convolution in space only
            return self.forward_no_time(x)

    def forward_no_time(self, x):
        """ x: (batch, channels, dim_x, dim_y)"""

        x0, x1 = x.size(2)//2 - self.modes1//2, x.size(2)//2 + self.modes1//2
        y0, y1 = x.size(3)//2 - self.modes2//2, x.size(3)//2 + self.modes2//2

        weights = torch.fft.ifftn(self.weights, dim=[-1], s=self.T)

        # Compute FFT of the input signal to convolve
        x_ft = torch.fft.fftn(x, dim=[2,3])
        x_ft = torch.fft.fftshift(x_ft, dim=[2,3])

        # Pointwise multiplication by complex matrix 
        out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), self.T, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, x0:x1, y0:y1, :] = compl_mul2d_time(x_ft[:, :, x0:x1, y0:y1], weights)
        # out_ft[:, :, x0:x1, y0:y1, :] = x_ft[:, :, x0:x1, y0:y1][...,None]*weights[None,...]

        # Compute Inverse FFT
        out_ft = torch.fft.ifftshift(out_ft, dim=[2,3])
        x = torch.fft.ifftn(out_ft, dim=[2,3], s=(x.size(2), x.size(3)))

        return x.real

###################
# Now we define the SPDEs solver
#
# We begin by defining the fixed point map.
###################
class IterationLayer(nn.Module):
    def __init__(self, spde_func, modes1, modes2, modes3, T):
        super(IterationLayer, self).__init__()
        """...
        """
        self._spde = spde_func
        self.convolution = KernelConvolution(spde_func._hidden_size, modes1, modes2, modes3, T)  

    def forward(self, u, xi):
        """ - u: (batch, hidden_channels, dim_x, dim_y, dim_t)
            - xi: (batch, forcing_channels, dim_x, dim_y, dim_t)
        """
        Fu, Gu = self._spde(u) 
        Guxi = torch.einsum('abcdef, acdef -> abdef', Gu, xi)
        sol = Fu + Guxi
        return self.convolution(sol)

###################
# Now we wrap it up into something that solves the SPDE.
###################
class NeuralFixedPoint(nn.Module):
    def __init__(self, spde_func, modes1, modes2, modes3, T, n_iter):
        super(NeuralFixedPoint, self).__init__()

        # self.padding = int(2**(np.ceil(np.log2(abs(2*T-1)))))

        self.n_iter = n_iter
        
        self.iter_layer = IterationLayer(spde_func, modes1, modes2, modes3, T) 
      
        self.initial_convolution = self.iter_layer.convolution

    def forward(self, z0, xi):
        """ - z0: (batch, hidden_channels, dim_x, dim_y)
            - xi: (batch, forcing_channels, dim_x, dim_y, dim_t)
        """
        
        # x = F.pad(x,[0,self.padding-10])
        z0 =  self.initial_convolution(z0, time=False) 

        z = z0
        for i in range(self.n_iter):
            y = z0 + self.iter_layer(z, xi)
            z = y
        
        return y