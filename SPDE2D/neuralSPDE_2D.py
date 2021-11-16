import torch
import torch.nn as nn
import torch.nn.functional as F
from .SPDE2Dint import NeuralFixedPoint 

class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        """ TODO: add possibility to have more layers 
        """

        model = [nn.Conv3d(in_size, out_size, 1), nn.BatchNorm3d(out_size), nn.Tanh()] 

        self._model = nn.Sequential(*model)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y, dim_t)"""
        return self._model(x)

###################
# Now we define the SPDEs.
#
# We begin by defining the generator SPDE.
###################

class SPDEFunc(torch.nn.Module):
    """ SPDE in functional form: partial_t u_t = F(u_t) + G(u_t) partial_t xi(t) """

    def __init__(self, noise_size, hidden_size):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        # F and G are resolution invariant MLP (acting on the channels). 
        self._F = MLP(hidden_size, hidden_size)  
        # if a non local operator is necessary use instead:
        # self._F = F_FNO(modes1, modes2, hidden_size, hidden_size, num_layers=2)
        self._G = MLP(hidden_size, hidden_size * noise_size)

    def forward(self, z):
        """ z: (batch, hidden_size, dim_x, dim_y, dim_t)"""

        # TODO: add possibility to add the space-time grid
        return self._F(z), self._G(z).view(z.size(0), self._hidden_size, self._noise_size, z.size(2), z.size(3), z.size(4))


###################
# Now we wrap it up into something that computes the SPDE.
###################

class NeuralSPDE(torch.nn.Module):  
    def __init__(self, data_size, noise_size, hidden_size, modes1, modes2, modes3, T, n_iter):
        super().__init__()
        """
        data size: the number of channels/coordinates of the solution u 
        noise_size: the number of channels/coordinates of the forcing xi
        hidden_size: the number of channels for z (in the latent space)
        modes1, modes2, T and n_iter: parameters for the integrator.
        """

        self._initial = nn.Linear(data_size, hidden_size)

        self._func = SPDEFunc(noise_size, hidden_size)

        readout = [nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, data_size)]
        self._readout = nn.Sequential(*readout)

        self._integrator = NeuralFixedPoint(self._func, modes1, modes2, modes3, T, n_iter=n_iter)

    def forward(self, u0, xi):
        """ u0: (batch, hidden_size, dim_x, dim_y)
            xi: (batch, hidden_size, dim_x, dim_y, dim_t)
        """

        # Actually solve the SPDE. 

        z0 = self._initial(u0.permute(0,2,3,1)).permute(0,3,1,2)

        zs = self._integrator(z0, xi)

        ys = self._readout(zs.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        
        return ys

###################
# Alternatively, define a generative model
###################

class Generator(torch.nn.Module):  
    def __init__(self, ic, wiener, data_size, initial_noise_size, noise_size, hidden_size, modes1, modes2, T, n_iter):
        super().__init__()
        
        self._wiener = wiener
        self._ic = ic

        self._initial = nn.Linear(initial_noise_size, hidden_size)
        self._func = SPDEFunc(noise_size, hidden_size)
        readout = [nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, data_size)]
        self._readout = nn.Sequential(*readout)
        self._integrator = NeuralFixedPoint(self._func, modes1, modes2, T, n_iter=n_iter)

    def forward(self, device, batch_size):

        # Sample and lift initial condition

        init_noise = self._ic.sample(num=batch_size, device=device)  #(batch, initial_noise_size, dim_x, dim_y)
        z0 = self._initial(init_noise.permute(0,2,3,1)).permute(0,3,1,2)

        # Sample Wiener process 

        xi = self._wiener.sample(batch_size, torch_device = device)
        xi = xi.unsqueeze(1)
        xi = torch.diff(xi,dim=-1)

        # Integrate and approximate fixed point

        zs = self._integrator(z0, xi)

        # Project back 

        ys = self._readout(zs.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        
        return ys

###################
# Next the discriminator. Here, we're going to use our SPDE model as the
# discriminator. Except that the forcing will not be random but the output of the generator instead.
# TODO: input the derivative instead. 
###################

class Discriminator(torch.nn.Module): 
    def __init__(self, data_size, hidden_size, modes1, modes2, modes3, T, n_iter):
        super().__init__()

        self._readin = nn.Linear(data_size, hidden_size)
        self._func = SPDEFunc(data_size, hidden_size)
        self._net =  NeuralFixedPoint(self._func, modes1, modes2, modes3, T, n_iter=n_iter)
        readout = [nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 1)]
        self._readout = nn.Sequential(*readout)

    def forward(self, x):
        """ - x: (batch, data_size, dim_x, dim_y, dim_t)
        """

        x0 = self._readin(x[...,0].permute(0,2,3,1)).permute(0,3,1,2)

        x = self._net(x0,x)

        x = self._readout(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)  # (batch, 1, dim_x, dim_y, dim_t)

        x = x[...,-1]          # (batch, 1, dim_x, dim_y)
 
        score = torch.sum(x,dim=[1,2,3])    # (batch)

        return score.mean()




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



###################
# A FNO module which can be used to model a non local operator F
# for F(u(t)) with u(t) a function of 2 space variables
###################

# To use within an SPDEFunc: 
# self._F = F_FNO(modes1, modes2, hidden_channels, hidden_channels, num_layers)

# class F_FNO(nn.Module):
#     def __init__(self, modes1, modes2, in_channels, hidden_channels, num_layers):
#         super(F_FNO, self).__init__()

#         """ an FNO to model F(u(t)) where u(t) is a function of 2 spatial variables """

#         self.modes1 = modes1
#         self.modes2 = modes2

#         self.fc0 = nn.Linear(in_channels, hidden_channels)

#         self.net = [ FNO_layer(modes1, modes2, hidden_channels) for i in range(num_layers-1) ]
#         self.net += [ FNO_layer(modes1, modes2, hidden_channels, last=True) ]
#         self.net = nn.Sequential(*self.net)

#         self.fc1 = nn.Linear(hidden_channels, 128)
#         self.fc2 = nn.Linear(128, in_channels)

#     def forward(self, x):
#         """ x: (batch, hidden_channels, dim_x, dim_y, dim_t)"""
        
#         x = x.permute(0, 2, 3, 4, 1)
#         x = self.fc0(x)
#         x = x.permute(0, 4, 1, 2, 3)

#         x = self.net(x)

#         x = x.permute(0, 2, 3, 4, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         x = x.permute(0, 4, 1, 2, 3)

#         return x

# ###################
# # A typical FNO layer
# ###################

# class FNO_layer(nn.Module):
#     def __init__(self, modes1, modes2, width, last=False):
#         super(FNO_layer, self).__init__()

#         self.last = last
#         self.conv = ConvolutionSpace(width, modes1, modes2)
#         self.w = nn.Linear(width, width)

#     def forward(self, x):
#         """ x: (batch, hidden_channels, dim_x, dim_y, dim_t)"""
#         x1 = self.conv(x)
#         x2 = self.w(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
#         x = x1 + x2
#         if not self.last:
#             x = F.gelu(x)           
#         return x


# ###################
# # Some helper functions to compute convolutions
# ###################

# def compl_mul2d_spatial(a, b):
#     """ ...
#     """
#     return torch.einsum("aibcd, ijbc -> ajbcd",a,b)

# class ConvolutionSpace(nn.Module):
#     def __init__(self, channels, modes1, modes2):
#         super(ConvolutionSpace, self).__init__()

#         """ ...    
#         """
#         self.modes1 = modes1
#         self.modes2 = modes2

#         self.scale = 1. / (channels**2)
#         self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, self.modes1, self.modes2, dtype=torch.cfloat))


#     def forward(self, x):
#         """ x: (batch, channels, dim_x, dim_y, dim_t)"""

#         x0, x1 = x.size(2)//2 - self.modes1//2, x.size(2)//2 + self.modes1//2
#         y0, y1 = x.size(3)//2 - self.modes2//2, x.size(3)//2 + self.modes2//2

#         # Compute FFT of the input signal to convolve
#         x_ft = torch.fft.fftn(x, dim=[2,3])
#         x_ft = torch.fft.fftshift(x_ft, dim=[2,3])

#         # Pointwise multiplication by complex matrix 
#         out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), x.size(4), device=x.device, dtype=torch.cfloat)
#         out_ft[:, :, x0:x1, y0:y1, :] = compl_mul2d_spatial(x_ft[:, :, x0:x1, y0:y1], self.weights)

#         # Compute Inverse FFT
#         out_ft = torch.fft.ifftshift(out_ft, dim=[2,3])
#         x = torch.fft.ifftn(out_ft, dim=[2,3], s=(x.size(2), x.size(3)))

#         return x.real