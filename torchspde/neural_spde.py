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

    def __init__(self, dim, in_channels, noise_channels, hidden_channels, modes1, modes2=None, modes3=None, n_iter=4, solver='fixed_point', **kwargs):
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