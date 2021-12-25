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
        op(A[0], z[:, 0]) - op(A[1], z[:, 1]),
        op(A[1], z[:, 0]) -op(A[0], z[:, 1])
    ], dim=1)

def compl_mat_vec_mul_2d(A, z):
    """A: contains complex matrices of coefficients  (2, dim_x, dim_y, hidden_size, hidden_size)
       z: (batch, 2, dim_x, dim_y, hidden_size)
       out: (batch, 2, dim_x, dim_y, hidden_size)
    """
    op = partial(torch.einsum, "xyij, bxyj-> bxyi") 

    return torch.stack([
        op(A[0], z[:, 0]) - op(A[1], z[:, 1]),
        op(A[1], z[:, 0]) -op(A[0], z[:, 1])
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
        super(ControlledODE, self).__init__() 
        
        scale = 1./(hidden_channels**2)

        self.flag1d = False if modes2 else True
        
        if self.flag1d:
            self.A = nn.Parameter(scale * torch.rand(2, modes1, hidden_channels, hidden_channels))
            self.modes = [modes1]
        else:
            self.A = nn.Parameter(scale * torch.rand(2, modes1, modes2, hidden_channels, hidden_channels)) 
            self.modes = [modes1, modes2]

        self.spde_func = spde_func

    def forward(self, t, v):
        """ v: (batch, 2, dim_x, (possibly dim y), hidden_size)"""

        freqs = [ (v.size(2+i)//2 - self.modes[i]//2, v.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]

        Av = torch.zeros(v.size(), device=v.device, dtype=v.dtype)

        if self.flag1d:
            Av[:, :, freqs[0][0]:freqs[0][1] ] = compl_mat_vec_mul_1d(self.A, v[:, :, freqs[0][0]:freqs[0][1] ] )
        else:
            Av[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ]  = compl_mat_vec_mul_2d(self.A, v[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] )

        return Av

    def prod(self, t, v, xi):
        # v is of shape (batch, 2, modes1, possibly modes2, hidden_channels) 
        # xi is of shape (batch, dim_x, possibly dim_y, noise_size)

        # lower and upper bounds of selected frequencies
        freqs = [ (v.size(2+i)//2 - self.modes[i]//2, v.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]
        out_size = v.size() 
        xi = xi[:,0,...] # we had to dupplicate xi so that its shape was compatible with the requirements of cdeint

        # compute Av
        Av = self.forward(t, v)

        # 1) FFT^-1
        if self.flag1d:
            dim_x = xi.size(2)
            v = torch.fft.ifftshift(v, dim=[2]) # centering modes
            v = torch.view_as_complex(v.permute(0,2,3,1).contiguous()) # (batch, modes1, hidden_channels) -- complex
            z = torch.fft.ifftn(v, dim=[1], s=dim_x).real.permute(0,2,1) # FFT^-1(v) (batch, hidden_channels, dim_x) -- real
        else:
            dim_x, dim_y = xi.size(2), xi.size(3)
            v = torch.fft.ifftshift(v, dim=[2, 3]) # centering modes
            v = torch.view_as_complex(v.permute(0,2,3,4,1).contiguous()) # (batch, modes1, modes2, hidden_channels) -- complex
            z = torch.fft.ifftn(v, dim=[1, 2], s=[dim_x, dim_y]).real.permute(0,3,1,2) # FFT^-1(v) (batch, hidden_channels, dim_x, dim_y) -- real

        # 2) H o FFT^-1
        
        # F_z is of shape (batch, hidden_channels, dim_x, possibly dim_y)
        # G_z is of shape (batch, hidden_channels, noise_channels, dim_x, possibly dim_y)
        
        F_z, G_z = self.spde_func(z) 
        
        if self.flag1d:
            G_z_xi = torch.einsum('bhnx, bxn -> bhx', G_z, xi) # Not sure...
        else:
            G_z_xi = torch.einsum('bhnxy, bxyn -> bhxy', G_z, xi) # Not sure...
        
        # H is of shape (batch, hidden_channels, dim_x, possibly dim_y)
        H = F_z + G_z_xi

        # 3) FFT o H o FFT^-1
        out_ft = torch.zeros(out_size, device=H.device, dtype=H.dtype)
        if self.flag1d:
            v = torch.fft.fftn(H, dim=[2]) # FFT(H) (batch, hidden_channels, dim_x) -- complex 
            v = torch.fft.fftshift(v, dim=[2]) # centering modes
            v = torch.stack([v.real, v.imag], dim=1) # (batch, 2, hidden_channels, dim_x) 
            v = v.permute(0,1,3,2)  # (batch, 2, dim_x, hidden_channels) 
            out_ft[:, :, :, freqs[0][0]:freqs[0][1] ]  = v[:, :, :, freqs[0][0]:freqs[0][1] ] 
        else:
            v = torch.fft.fftn(H, dim=[2,3]) # FFT(H) (batch, hidden_channels, dim_x, dim_y) -- complex 
            v = torch.fft.fftshift(v, dim=[2,3]) # centering modes
            v = torch.stack([v.real, v.imag], dim=1) # (batch, 2, hidden_channels, dim_x, dim_y) 
            v = v.permute(0,1,3,4,2)  # (batch, 2, dim_x, dim_y, hidden_channels) 
            out_ft[:, :, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ]  = v[:, :, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] 

        # We form the vector field A + FFT o H o FFT^-1
        sol = Av + out_ft
       
        return sol


#=============================================================================================
# SPDE solver: linear controlled differential equation solver in Fourier space.
#=============================================================================================

class DiffeqSolver(nn.Module):
    def __init__(self, hidden_channels, spde_func, modes1, modes2=None, **kwargs):
        super(DiffeqSolver, self).__init__()

        self.spde_func = spde_func
        
        self.cde = ControlledODE(spde_func, hidden_channels, modes1, modes2)

        self.flag1d = False if modes2 else True
        if self.flag1d:
            self.dims = [2]
            self.modes = [modes1]
        else:
            self.dims = [2,3] 
            self.modes = [modes1, modes2]

        if 'adjoint' not in kwargs:
            kwargs['adjoint']=False
        self.kwargs = kwargs

    def forward(self, z0, xi, grid=None):
        """ - z0: (batch, hidden_channels, dim_x, (possibly dim_y))
            - xi: (batch, forcing_channels, dim_x, (possibly dim_y), dim_t)
            - grid: should be speficied if computing gradients of the solution
        """

        # lower and upper bounds of selected frequencies
        freqs = [ (z0.size(2+i)//2 - self.modes[i]//2, z0.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]

        # compute fourier transform of initial condition  
        z0_ft = torch.fft.fftshift(torch.fft.fftn(z0, dim=self.dims), dim=self.dims) 
        z0_ft = torch.stack([z0_ft.real, z0_ft.imag], dim=1) # (batch, 2, hidden_channels, dim_x, possibly dim_y)

        # antialiasing (the highest modes are set to zero)  # TODO: would padding be more efficient?
        v0 = torch.zeros(z0_ft.size(), device=z0.device, dtype=z0.dtype)
        if self.flag1d:
            v0[:, :, :, freqs[0][0]:freqs[0][1] ]  = z0_ft[:, :, :, freqs[0][0]:freqs[0][1] ] 
        else:
            v0[:, :, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ]  = z0_ft[:, :, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] 
        
        # reshape for cdeint 
        if self.flag1d:
            v0 = v0.permute(0,1,3,2) # (batch, 2, dim_x, hidden_channels)
            xi = xi.permute(0,2,3,1) # (batch, dim_x, dim_t, hidden_channels)
        else:
            v0 = v0.permute(0,1,3,4,2) # (batch, 2, dim_x, dim_y, hidden_channels)
            xi = xi.permute(0,2,3,4,1) # (batch, dim_x, dim_y, dim_t, hidden_channels)

        # hack so that xi's shape is compatible with that of v0
        xi = torch.stack([xi, xi], dim=1) # (batch,2, dim_x, possibly dim_y, dim_t, hidden_channels)

        # interpolate xi so that it can be queried at any time t 
        xi = torchcde.linear_interpolation_coeffs(xi)
        xi = LinearInterpolation(xi) 
  
        # Solve the CDE,  get v of shape (batch, 2, dim_x, (possibly dim_y), dim_t, hidden_channels) 
        v = torchcde.cdeint(X=xi,
                            z0=v0,
                            func=self.cde,
                            method='euler',
                            t=xi._t,
                            # adjoint = self.kwargs['adjoint'],
                            **self.kwargs) 

        # Compute z = FFT^-1(v) 
        if self.flag1d:
            v = v.permute(0,4,2,3,1) # (batch, hidden_channels, dim_x, dim_t, 2) 
        else:
            v = v.permute(0,5,2,3,4,1) # (batch, hidden_channels, dim_x, dim_y, dim_t, 2) 

        v = torch.view_as_complex(v.contiguous()) # (batch, hidden_channels, dim_x, dim_y, dim_t) -- complex 

        z = torch.fft.ifftn(torch.fft.ifftshift(v, dim=self.dims), dim=self.dims).real  # (batch, hidden_channels, dim_x, dim_y, dim_t) -- real 

        return z  # (batch, hidden_channels, dim_x, dim_t)
