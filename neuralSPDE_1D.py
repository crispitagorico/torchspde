import torch
import torch.nn as nn
import torch.nn.functional as F
from SPDE1Dint import NeuralFixedPoint

class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        """ TODO: add possibility to have more layers 
        """

        model = [nn.Conv2d(in_size, out_size, 1), nn.BatchNorm2d(out_size), nn.Tanh()] 

        self._model = nn.Sequential(*model)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""
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
        self._G = MLP(hidden_size, hidden_size * noise_size)

    def forward(self, z):
        """ z: (batch, hidden_size, dim_x, dim_t)"""

        # TODO: add possibility to add the space-time grid
        return self._F(z), self._G(z).view(z.size(0), self._hidden_size, self._noise_size, z.size(2), z.size(3))


###################
# Now we wrap it up into something that computes the SPDE.
###################

class NeuralSPDE(torch.nn.Module):  
    def __init__(self, data_size, noise_size, hidden_size, modes1, modes2, T, n_iter):
        super().__init__()
        """
        data size: the number of channels/coordinates of the solution u 
        noise_size: the number of channels/coordinates of the forcing xi
        hidden_size: the number of channels for z (in the latent space)
        modes1, modes2, T and n_iter: parameters for the integrator.
        """

        self._initial = nn.Linear(data_size, hidden_size)

        self._func = GeneratorFunc(noise_size, hidden_size)

        readout = [nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, data_size)]
        self._readout = nn.Sequential(*readout)

        self._integrator = NeuralFixedPoint(self._func, modes1, modes2, T, n_iter=n_iter)

    def forward(self, u0, xi):
        """ u0: (batch, hidden_size, dim_x)
            xi: (batch, hidden_size, dim_x, dim_t)
        """

        # Actually solve the SPDE. 

        z0 = self._initial(u0.permute(0,2,1)).permute(0,2,1)

        zs = self._integrator(z0, xi)

        ys = self._readout(zs.permute(0,2,3,1)).permute(0,3,1,2)
        
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

        init_noise = self._ic.sample(num=batch_size, device=device)
        z0 = self._initial(init_noise.permute(0,2,1)).permute(0,2,1)

        # Sample Wiener process 

        xi = self._wiener.sample(batch_size, torch_device = device)
        xi = xi.unsqueeze(1)
        xi = torch.diff(xi,dim=-1)

        # Integrate and approximate fixed point

        zs = self._integrator(z0, xi)

        # Project back 

        ys = self._readout(zs.permute(0,2,3,1)).permute(0,3,1,2)
        
        return ys

###################
# Next the discriminator. Here, we're going to use our SPDE model as the
# discriminator. Except that the forcing will not be random but the output of the generator instead.
# TODO: input the derivative instead. 
###################

class Discriminator(torch.nn.Module): 
    def __init__(self, data_size, hidden_size, modes1, modes2, T, n_iter):
        super().__init__()

        self._readin = nn.Linear(data_size, hidden_size)
        self._func = SPDEFunc(data_size, hidden_size)
        self._net =  NeuralFixedPoint(self._func, modes1, modes2, T, n_iter=n_iter)
        readout = [nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, data_size)]
        self._readout = nn.Sequential(*readout)

    def forward(self, x):
        """ - x: (batch, data_size, dim_x, dim_t)
        """

        x0 = self._readin(x[...,0].permute(0,2,1)).permute(0,2,1)

        x = self._net(x0,x)

        x = self._readout(x.permute(0,2,3,1)).permute(0,3,1,2)  # (batch, 1, dim_x, dim_t)

        x = x[...,-1]          # (batch, 1, dim_x)
 
        score = torch.sum(x,dim=[1,2])    # (batch)

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