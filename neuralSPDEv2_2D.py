import torch
import torch.nn as nn
import torch.nn.functional as F


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


class G(nn.Module):
    def __init__(self, hidden_channels, forcing_channels):
        super(G, self).__init__()
        """ ...
        """
        self.forcing_channels = forcing_channels

        # net = [nn.Linear(hidden_channels, hidden_channels*forcing_channels), nn.Tanh()]
        net = [nn.Conv3d(hidden_channels, hidden_channels*forcing_channels, 1), nn.BatchNorm3d(hidden_channels*forcing_channels), nn.Tanh()] 
        self.net = nn.Sequential(*net)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y, dim_t)"""
        return self.net(x).view(x.size(0), x.size(1), self.forcing_channels, x.size(2), x.size(3), x.size(4))


class IterationLayer(nn.Module):
    def __init__(self, modes1, modes2, modes3, hidden_channels, forcing_channels, T, F_depth):
        super(IterationLayer, self).__init__()
        """...
        """
        self.F = F_(modes1, modes2, hidden_channels, hidden_channels, F_depth)
        self.G = G(hidden_channels, forcing_channels)
        self.convolution = KernelConvolution(hidden_channels, modes1, modes2, modes3, T)

    def forward(self, x, xi):
        """ - x: (batch, hidden_channels, dim_x, dim_y, dim_t)
            - xi: (batch, forcing_channels, dim_x, dim_y, dim_t)
        """
        x_f = self.F(x) 
        mat = self.G(x) 
        x_g = torch.einsum('abcdef, acdef -> abdef', mat, xi)
        y = x_f + x_g
        return self.convolution(y)
        

class NeuralFixedPoint(nn.Module):
    def __init__(self, modes1, modes2, modes3, in_channels, hidden_channels, forcing_channels, out_channels, T, F_depth, n_iter):
        super(NeuralFixedPoint, self).__init__()

        """ ...
        """

        # self.padding = int(2**(np.ceil(np.log2(abs(2*T-1)))))

        self.n_iter = n_iter
        
        self.readin = nn.Linear(in_channels, hidden_channels)
        
        self.iter_layer = IterationLayer(modes1, modes2, modes3, hidden_channels, forcing_channels, T, F_depth) 
      
        self.initial_convolution = self.iter_layer.convolution

        readout = [nn.Linear(hidden_channels, 128), nn.ReLU(), nn.Linear(128, out_channels)]
        self.readout = nn.Sequential(*readout)

    def forward(self, x, xi):
        """ - x: (batch, in_channels, dim_x, dim_y)
            - xi: (batch, forcing_channels, dim_x, dim_y, dim_t)
        """
        
        z0 = self.readin(x.permute(0,2,3,1)).permute(0,3,1,2)
        
        # x = F.pad(x,[0,self.padding-10])
        z0 =  self.initial_convolution(z0, time=False) 

        z = z0
        for i in range(self.n_iter):
            y = z0 + self.iter_layer(z, xi)
            z = y
        
        return self.readout(y.permute(0,2,3,4,1)).permute(0,4,1,2,3)



''' To model non local F(u(t)) with u(t) a function of 2 space variables'''

class F_(nn.Module):
    def __init__(self, modes1, modes2, width, width2, L):
        super(F_, self).__init__()

        """ an FNO to model F(u(t)) where u(t) is a function of 2 spatial variables """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(self.width, width2)

        self.net = [ FNO_layer(modes1, modes2, width2) for i in range(L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width2, last=True) ]
        self.net = nn.Sequential(*self.net)

        self.fc1 = nn.Linear(width2, 128)
        self.fc2 = nn.Linear(128, self.width)

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

class FNO_layer(nn.Module):
    def __init__(self, modes1, modes2, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
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