import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchcde 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .linear_interpolation import LinearInterpolation


#=============================================================================================
# Convolution in physical space = pointwise mutliplication of complex tensors in Fourier space
#=============================================================================================

def compl_mat_vec_mul_1d(A, z):
    """A: contains complex matrices of coefficients  (2, dim_x, hidden_size, hidden_size)
       z: (batch, 2, dim_x, hidden_size)
       out: (batch, 2, dim_x, hidden_size)
    """
    op = partial(torch.einsum, "xij, bxj-> bxi") 

    return torch.stack([
        op(P[0], z[:, 0]) - op(P[1], z[:, 1]),
        op(P[1], z[:, 0]) -op(P[0], z[:, 1])
    ], dim=1)

def compl_mat_vec_mul_2d(A, z):
    """A: contains complex matrices of coefficients  (2, dim_x, dim_y, hidden_size, hidden_size)
       z: (batch, 2, dim_x, dim_y, hidden_size)
       out: (batch, 2, dim_x, dim_y, hidden_size)
    """
    op = partial(torch.einsum, "xyij, bxyj-> bxyi") 

    return torch.stack([
        op(P[0], z[:, 0]) - op(P[1], z[:, 1]),
        op(P[1], z[:, 0]) -op(P[0], z[:, 1])
    ], dim=1)



#=============================================================================================
# Non-linear controlled ODE
#=============================================================================================

class ControlledODE(torch.nn.Module):
    """Differential Equation solver in Fourier space: R'(t) = A*R(t) + control.
       A is a complex matrix resulting from a prior p;
       control is the space Fourier transform of H (see paper).
    """

    def __init__(self, spde_func, hidden_channels, modes1, modes2=None):
        super(ControlledODE).__init__() 
        
        scale = 1./(hidden_channels**2)

        self.flag1d = False if modes2 else True
        
        if self.flag1d:
            self.A = nn.Parameter(scale * torch.rand(2, modes1, hidden_channels, hidden_channels))
        else:
            self.A = nn.Parameter(scale * torch.rand(2, modes1, modes2, hidden_channels, hidden_channels)) 
        
        self.spde_func = spde_func

    def forward(self, t, z):
        """ z: (batch, 2, dim_x, (possibly dim y), hidden_size)"""
        
        if self.flag1d:
            Az = compl_mat_vec_mul_1d(self.A, z)
        else:
            Az = compl_mat_vec_mul_2d(self.A, z)

        return Az

    def prod(self, t, v, xi):
        # v is of shape (N, 2, modes1, possibly modes2, hidden_channels) 
        # xi is of shape (N, 2, dim_x, possibly dim_y, noise_size)

        if self.flag1d:
            dim_x = xi.size(2)
            z = torch.fft.ifftn(torch.fft.ifftshift(v, dim=[-1]), dim=[-1], s=z0_path.size(-1))
        else:
            dim_x, dim_y = xi.size(2), xi.size(3)
        

        F_z, G_z = self.spde_func(z) 

        if self.flag1d:
            G_z_xi = torch.einsum('abcde, acde -> abde', G_z, xi)
        else:
            G_z_xi = torch.einsum('abcdef, acdef -> abdef', G_z, xi)

        H_z_xi = F_z + G_z_xi
        
        Pz = self.forward(t, z)

        sol = Pz + control_gradient
       
        return sol


#=============================================================================================
# SPDE solver: linear controlled differential equation solver in Fourier space.
#=============================================================================================

class FourierCDE(nn.Module):
    def __init__(self, hidden_channels, spde_func, modes1, modes2=None):
        super(FourierCDE, self).__init__()

        self.spde_func = spde_func
        
        self.cde = LinearCDE(hidden_channels, modes1, modes2)

        self.flag1d = False if modes2 else True
        if self.flag1d:
            self.dims = [2]
        else:
            self.dims = [2,3]

    def forward(self, z0, xi):
        """ - u0: (batch, hidden_channels, dim_x, (possibly dim_y))
            - xi: (batch, forcing_channels, dim_x, (possibly dim_y), dim_t)
        """

        # compute fourier transform of initial condition
        z0_ft = torch.fft.fftshift(torch.fft.fftn(z0, dim=self.dims), dim=self.dims)

        # create control of CDE in Fourier space
        Fu, Gu = self._spde_rhs(u) 
        Guxi = torch.einsum('bijxt, bjxt -> bixt', Gu, xi)
        Hu = Fu + Guxi 

        H_ft = torch.fft.fftshift(torch.fft.fftn(Hu, dim=[2]), dim=[2])  #(batch, hidden_channels, dim_x, dim_t) 

        # Now we prepare the data for cdeint  
        
        H_ft = H_ft.permute(0, 2, 3, 1)  # (batch, dim_x, dim_t, hidden_channels)
 
        H_ft = torch.stack([H_ft.real, H_ft.imag], dim=1)  # (batch, 2, dim_x, dim_t, hidden_channels)

        H_ft = torchcde.linear_interpolation_coeffs(H_ft)

        H_ft = LinearInterpolation(H_ft)

        u0_ft = u0_ft.permute(0, 2, 1)   # (batch, dim_x, hidden_channels)

        u0_ft = torch.stack([u0_ft.real, u0_ft.imag], dim=1)  # (batch, 2, dim_x, hidden_channels)
  
        # Solve the CDE
        u_new_ft = torchcde.cdeint(X=H_ft,
                                   z0=u0_ft,
                                   func=self._linear_func,
                                   method='euler',
                                   t=H_ft._t) 

        # u_new_ft is (batch, 2, dim_x, dim_t, hidden_channels)  
        u_new_ft = u_new_ft.permute(0, 4, 2, 3, 1) # (batch, hidden_channels, dim_x, 2)

        u_new_ft = torch.view_as_complex(u_new_ft.contiguous()) # (batch, hidden_channels, dim_x, dim_t)

        u_new = torch.fft.ifftn(torch.fft.ifftshift(u_new_ft, dim=[2]), dim=[2])

        return u_new.real  # (batch, hidden_channels, dim_x, dim_t)
