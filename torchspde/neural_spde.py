import torch
import torch.nn as nn
import torch.nn.functional as F
from .fixed_point_solver import NeuralFixedPoint 
from .diffeq_solver import DiffeqSolver

class SPDEFunc0d(torch.nn.Module):
    """ Modelling local operators F and G in (latent) SPDE (d_t - L)u = F(u)dt + G(u) dxi_t 
    """

    def __init__(self, noise_channels, hidden_channels):
        """hidden_channels is d_h in the paper
           noise_channels is d_xi in the paper 
        """
        super().__init__()
        self.noise_channels = noise_channels
        self.hidden_channels = hidden_channels

        # local non-linearity F
        model_F = [nn.Conv1d(hidden_channels, hidden_channels, 1), nn.BatchNorm1d(hidden_channels), nn.Tanh()]
        self.F = nn.Sequential(*model_F)  

        # local non-linearity G
        model_G = [nn.Conv1d(hidden_channels, hidden_channels*noise_channels, 1), nn.BatchNorm1d(hidden_channels*noise_channels), nn.Tanh()]  
        self.G = nn.Sequential(*model_G)

    def forward(self, z):
        """ z: (batch, hidden_channels, dim_x)
        """

        # TODO: add possibility to add the space-time grid
        return self.F(z), self.G(z).view(z.size(0), self.hidden_channels, self.noise_channels, z.size(2))


class SPDEFunc1d(torch.nn.Module):
    """ Modelling local operators F and G in (latent) SPDE (d_t - L)u = F(u)dt + G(u) dxi_t 
    """

    def __init__(self, noise_channels, hidden_channels):
        """hidden_channels is d_h in the paper
           noise_channels is d_xi in the paper 
        """
        super().__init__()
        self.noise_channels = noise_channels
        self.hidden_channels = hidden_channels

        # local non-linearity F
        model_F = [nn.Conv2d(hidden_channels, hidden_channels, 1), nn.BatchNorm2d(hidden_channels), nn.Tanh()]
        self.F = nn.Sequential(*model_F)  

        # local non-linearity G
        model_G = [nn.Conv2d(hidden_channels, hidden_channels*noise_channels, 1), nn.BatchNorm2d(hidden_channels*noise_channels), nn.Tanh()]  
        self.G = nn.Sequential(*model_G)

    def forward(self, z):
        """ z: (batch, hidden_channels, dim_x, dim_t)
        """

        # TODO: add possibility to add the space-time grid
        return self.F(z), self.G(z).view(z.size(0), self.hidden_channels, self.noise_channels, z.size(2), z.size(3))


class SPDEFunc2d(torch.nn.Module):
    """ Modelling local operators F and G in (latent) SPDE (d_t - L)u = F(u)dt + G(u) dxi_t 
    """

    def __init__(self, noise_channels, hidden_channels):
        """hidden_channels is d_h in the paper
           noise_channels is d_xi in the paper 
        """
        super().__init__()
        self.noise_channels = noise_channels
        self.hidden_channels = hidden_channels

        # local non-linearity F 
        model_F = [nn.Conv3d(hidden_channels, hidden_channels, 1), nn.BatchNorm3d(hidden_channels), nn.Tanh()]
        self.F = nn.Sequential(*model_F)  

        # local non-linearity G
        model_G = [nn.Conv3d(hidden_channels, hidden_channels*noise_channels, 1), nn.BatchNorm3d(hidden_channels*noise_channels), nn.Tanh()]  
        self.G = nn.Sequential(*model_G)

    def forward(self, z):
        """ z: (batch, hidden_channels, dim_x, dim_y, dim_t)
        """

        # TODO: add possibility to add the space-time grid
        return self.F(z), self.G(z).view(z.size(0), self.hidden_channels, self.noise_channels, z.size(2), z.size(3), z.size(4))


class NeuralSPDE(torch.nn.Module):  

    def __init__(self, dim, in_channels, noise_channels, hidden_channels, n_iter, modes1, modes2=None, modes3=None, solver='fixed_point', **kwargs):
        super().__init__()
        """
        dim: dimension of spatial domain (1 or 2 for now)
        in_channels: the dimension of the solution state space
        noise_channels: the dimension of the control state space
        hidden_channels: the dimension of the latent space
        modes1, modes2, (possibly modes 3): Fourier modes
        solver: 'fixed_point' or 'diffeq'
        kwargs: Any additional kwargs to pass to the cdeint solver of torchdiffeq
        """

        assert dim in [1,2], 'dimension of spatial domain (1 or 2 for now)'
        if dim == 2 and solver == 'fixed_point':
            assert modes2 is not None and modes3 is not None, 'specify modes2 and modes3' 
        if dim == 1 and solver == 'fixed_point':
            assert modes2 is not None and modes3 is None, 'specify modes2 and modes3 should not be specified' 
        if dim == 2 and solver == 'diffeq':
            assert modes2 is not None, 'specify modes2' 
        if dim == 1 and solver == 'diffeq':
            assert modes2 is None, 'modes2 should not be specified' 

        self.dim = dim

        # initial lift
        self.lift = nn.Linear(in_channels, hidden_channels)

        if dim==1 and solver=='diffeq':
            self.spde_func = SPDEFunc0d(noise_channels, hidden_channels)
        if (dim==1 and solver=='fixed_point') or (dim==2 and solver=='diffeq'):
            self.spde_func = SPDEFunc1d(noise_channels, hidden_channels)
        if (dim==2 and solver=='fixed_point'):
            self.spde_func = SPDEFunc2d(noise_channels, hidden_channels)

        # linear projection
        readout = [nn.Linear(hidden_channels, 128), nn.ReLU(), nn.Linear(128, in_channels)]
        self.readout = nn.Sequential(*readout)

        # SPDE solver (for now Picard)
        if solver=='fixed_point':
            self.solver = NeuralFixedPoint(self.spde_func, n_iter, modes1, modes2, modes3)
        elif solver=='diffeq':
            self.solver = DiffeqSolver(hidden_channels, self.spde_func, modes1, modes2, **kwargs)


    def forward(self, u0, xi, grid=None):
        """ u0: (batch, hidden_size, dim_x, (possibly dim_y))
            xi: (batch, hidden_size, dim_x, (possibly dim_y), dim_t)
            grid: (batch, dim_x, (possibly dim_y), dim_t)
        """
        if grid is not None:
            grid = grid[0]
            
        # Actually solve the SPDE. 
        if self.dim==1:
            z0 = self.lift(u0.permute(0,2,1)).permute(0,2,1) 
        else:
            z0 = self.lift(u0.permute(0,2,3,1)).permute(0,3,1,2)

        zs = self.solver(z0, xi, grid)

        if self.dim==1:
            ys = self.readout(zs.permute(0,2,3,1)).permute(0,3,1,2)
        else:
            ys = self.readout(zs.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        
        return ys











###################
# Alternatively, define a generative model
###################

#class Generator(torch.nn.Module):  
#    def __init__(self, ic, wiener, data_size, initial_noise_size, noise_size, hidden_size, modes1, modes2, T, n_iter):
#        super().__init__()
        
#        self._wiener = wiener
#        self._ic = ic

#        self._initial = nn.Linear(initial_noise_size, hidden_size)
#        self._func = SPDEFunc(noise_size, hidden_size)
#        readout = [nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, data_size)]
#        self._readout = nn.Sequential(*readout)
#        self._integrator = NeuralFixedPoint(self._func, modes1, modes2, T, n_iter=n_iter)

#    def forward(self, device, batch_size):

#        # Sample and lift initial condition

#        init_noise = self._ic.sample(num=batch_size, device=device)
#        z0 = self._initial(init_noise.permute(0,2,1)).permute(0,2,1)

#        # Sample Wiener process 

#        xi = self._wiener.sample(batch_size, torch_device = device)
#        xi = xi.unsqueeze(1)
#        xi = torch.diff(xi,dim=-1)

#        # Integrate and approximate fixed point

#        zs = self._integrator(z0, xi)

#        # Project back 

#        ys = self._readout(zs.permute(0,2,3,1)).permute(0,3,1,2)
        
#        return ys

####################
## Next the discriminator. Here, we're going to use our SPDE model as the
## discriminator. Except that the forcing will not be random but the output of the generator instead.
## TODO: input the derivative instead. 
####################

#class Discriminator(torch.nn.Module): 
#    def __init__(self, data_size, hidden_size, modes1, modes2, T, n_iter):
#        super().__init__()

#        self._readin = nn.Linear(data_size, hidden_size)
#        self._func = SPDEFunc(data_size, hidden_size)
#        self._net =  NeuralFixedPoint(self._func, modes1, modes2, T, n_iter=n_iter)
#        readout = [nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, data_size)]
#        self._readout = nn.Sequential(*readout)

#    def forward(self, x):
#        """ - x: (batch, data_size, dim_x, dim_t)
#        """

#        x0 = self._readin(x[...,0].permute(0,2,1)).permute(0,2,1)

#        x = self._net(x0,x)

#        x = self._readout(x.permute(0,2,3,1)).permute(0,3,1,2)  # (batch, 1, dim_x, dim_t)

#        x = x[...,-1]          # (batch, 1, dim_x)
 
#        score = torch.sum(x,dim=[1,2])    # (batch)

#        return score.mean()

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