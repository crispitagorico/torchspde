import torch
import torch.nn as nn
import torch.nn.functional as F


def compl_mul2d(a, b):
    """ ...
    """
    return torch.einsum("aibc, ijbc -> ajbc",a,b)


def compl_mul1d_time(a, b):
    """ ...
    """
    return torch.einsum("aib, ijbc -> ajbc",a,b)


class KernelConvolution(nn.Module):
    def __init__(self, channels, modes1, modes2, T):
        super(KernelConvolution, self).__init__()

        """ ...    
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.T = T

        self.scale = 1. / (channels**2)
        self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes1, self.modes2, dtype=torch.cfloat))
        # self.weights = nn.Parameter(self.scale * torch.rand(channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
    def forward(self, x, time=True):
        """ x: (batch, channels, dim_x, dim_y, dim_t)"""

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

        weights = torch.fft.ifftn(self.weights, dim=[-1], s=self.T)

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


class F(nn.Module):
    def __init__(self, hidden_channels, forcing_channels):
        super(F, self).__init__()
        """ ...
        """
        self.forcing_channels = forcing_channels

        # net = [nn.Linear(hidden_channels, hidden_channels*forcing_channels), nn.Tanh()]
        net = [nn.Conv2d(hidden_channels, hidden_channels*forcing_channels, 1), nn.BatchNorm2d(hidden_channels*forcing_channels), nn.Tanh()] 
        self.net = nn.Sequential(*net)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""
        return self.net(x).view(x.size(0), x.size(1), self.forcing_channels, x.size(2), x.size(3))


class IterationLayer(nn.Module):
    def __init__(self, modes1, modes2, hidden_channels, forcing_channels, T):
        super(IterationLayer, self).__init__()
        """...
        """

        self.F = F(hidden_channels, forcing_channels)
        self.convolution = KernelConvolution(hidden_channels, modes1, modes2, T)

    def forward(self, x, xi):
        """ - x: (batch, hidden_channels, dim_x, dim_t)
            - xi: (batch, forcing_channels, dim_x, dim_t)
        """
        mat = self.F(x)
        y = torch.einsum('abcde, acde -> abde', mat, xi)
        return self.convolution(y)
        

class NeuralFixedPoint(nn.Module):
    def __init__(self, modes1, modes2, in_channels, hidden_channels, forcing_channels, out_channels, T, n_iter):
        super(NeuralFixedPoint, self).__init__()

        """ ...
        """

        # self.padding = int(2**(np.ceil(np.log2(abs(2*T-1)))))

        self.n_iter = n_iter
        
        self.readin = nn.Linear(in_channels, hidden_channels)
        
        self.iter_layer = IterationLayer(modes1, modes2, hidden_channels, forcing_channels, T) 
      
        self.initial_convolution = self.iter_layer.convolution

        readout = [nn.Linear(hidden_channels, 128), nn.ReLU(), nn.Linear(128, out_channels)]
        self.readout = nn.Sequential(*readout)

    def forward(self, x, xi):
        """ - x: (batch, in_channels, dim_x)
            - xi: (batch, forcing_channels, dim_x, dim_t)
        """

        z0 = self.readin(x.permute(0,2,1)).permute(0,2,1)
        
        # x = F.pad(x,[0,self.padding-10])
        z0 =  self.initial_convolution(z0, time=False) 

        z = z0
        for i in range(self.n_iter):
            y = z0 + self.iter_layer(z, xi)
            z = y
        
        return self.readout(y.permute(0,2,3,1)).permute(0,3,1,2)






# class AddDerivatives(nn.Module):
#     def __init__(self, order):
#         super(AddDerivatives, self).__init__()
#         """
#         Class to augment a function f(x,y,t) with its derivatives wrt x,y up to order.    
#         """
#         self.order = order

#     def forward(self, x):
#         """ x: (batch, channels, dim_x, dim_y, dim_t)"""

#         L_x, L_y = x.size(2), x.size(3)

#         # compute Fourier frequencies
#         k_x = (2.*np.pi)*L_x*torch.fft.fftfreq(L_x).repeat(L_x,1).transpose(0,1).to(device) 
#         k_y = (2.*np.pi)*L_y*torch.fft.fftfreq(L_y).repeat(L_y,1).to(device)

#         # compute FFT (2D in space because we only consider space derivatives)
#         x_ft = torch.fft.fft2(x, dim=[2,3])

#         # compute derivatives
#         derivatives = x_ft
#         for k in range(self.order):

#             x_ft_x = 1j * k_x[...,None]*x_ft
#             x_ft_y = 1j * k_y[...,None]*x_ft

#             # store derivatives
#             x_ft = torch.cat([x_ft_x,x_ft_y],dim=1)
#             derivatives = torch.cat([derivatives, x_ft], dim=1)

#         # go back to physical domain
#         # derivatives = torch.view_as_complex(derivatives)
#         return torch.fft.ifft2(derivatives, dim=[2,3], s=(x.size(2), x.size(3))).real



# class PolynomialFeatures(nn.Module):
#     def __init__(self, degree):
#         super(PolynomialFeatures, self).__init__()
#         """Computes polynomial features up to degree (>0)"""
#         assert degree in [1,2], 'currently only implemented for polynomial features of degree 1 and 2'
#         self.degree = degree

#     def forward(self,x):
#       """ x: (batch, channels, 64, 64, 10)"""      
#       if self.degree==2:
#           monomials = torch.einsum("""aibcd, ajbcd -> aijbcd """, x, x).view(x.size(0), x.size(1)**2, x.size(2), x.size(3), x.size(4))
#           return torch.cat([x, monomials], dim=1)
#       return x


# def compute_nb_new_channels(channels, order, degree):
#     nb_derivatives = 2**(order+1)-1
#     channels *= nb_derivatives
#     # if degree==2:
#     #     channels += channels**2
#     return channels