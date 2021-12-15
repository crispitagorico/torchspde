import torch
import torch.nn as nn
import torch.nn.functional as F

#=============================================================================================
# Convolution in physical space = pointwise mutliplication of complex tensors in Fourier space
#=============================================================================================

def compl_mul2d(func_fft, kernel_tensor):
    return torch.einsum("bixt, ijxt -> bjxt", func_fft, kernel_tensor)


def compl_mul1d_time(func_fft, kernel_tensor):
    return torch.einsum("bixt, ijxt -> bjxt", func_fft, kernel_tensor)


def compl_mul3d(func_fft, kernel_tensor):
    return torch.einsum("bixyt, ijxyt -> bjxyt", func_fft, kernel_tensor)


def compl_mul2d_time(func_fft, kernel_tensor):
    return torch.einsum("bixyt, ijxyt -> bjxyt", func_fft, kernel_tensor)


#=============================================================================================
# Semigroup action is integration against a kernel
#=============================================================================================
class KernelConvolution(nn.Module):
    def __init__(self, channels, modes1, modes2, modes3=None):
        super(KernelConvolution, self).__init__()

        """ This module has a kernel parametrized as a complex tensor in the spectral domain. 
            The method forward computes S*H;
            The method forward_init computes S_t*u_0.
        """

        self.scale = 1. / (channels**2)

        # define kernel-tensor shape depending on dim of the problem
        if not modes3: # 1d
            self.modes = [modes1, modes2]
            self.dims = [2,3]
            self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, modes1, modes2,  dtype=torch.cfloat)) # K_theta in paper
        else: # 2d
            self.modes = [modes1, modes2, modes3]
            self.dims = [2,3,4]
            self.weights = nn.Parameter(self.scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat)) # K_theta in paper
       
   
    def forward(self, z, init=False):
        """ z: (batch, channels, dim_x, (possibly dim_y), dim_t)"""
        
        # lower and upper bounds of selected frequencies
        freqs = [ (z.size(2+i)//2 - self.modes[i]//2, z.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)) ]
 
        if not init: # S * u

            # Compute FFT
            z_ft = torch.fft.fftn(z, dim=self.dims)
            z_ft = torch.fft.fftshift(z_ft, dim=self.dims)
 
            # Pointwise multiplication of kernel_tensor and func_fft
            out_ft = torch.zeros(z.size(), device=z.device, dtype=torch.cfloat)
            if len(self.modes)==2: # 1d case
                out_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ] = compl_mul2d(z_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ], self.weights)
            else: # 2d case
                out_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1], freqs[2][0]:freqs[2][1] ] = compl_mul3d(z_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1], freqs[2][0]:freqs[2][1] ], self.weights)
            
            # Compute Inverse FFT
            out_ft = torch.fft.ifftshift(out_ft, dim=self.dims)
            z = torch.fft.ifftn(out_ft, dim=self.dims)
            return z.real

        else: # S_t * z_0
            return self.forward_init(z)

    
    def forward_init(self, z0_path):
        """ z0_path: (batch, channels, dim_x, dim_t)"""

        # lower and upper bounds of selected frequencies
        freqs = [ (z0_path.size(2+i)//2 - self.modes[i]//2, z0_path.size(2+i)//2 + self.modes[i]//2) for i in range(len(self.modes)-1) ]

        # K_t = F_t^-1(K)
        weights = torch.fft.ifftn(torch.fft.ifftshift(self.weights, dim=[-1]), dim=[-1], s=z0_path.size(-1))

        # Compute FFT of the input signal to convolve
        z_ft = torch.fft.fftn(z0_path, dim=self.dims[:-1])
        z_ft = torch.fft.fftshift(z_ft, dim=self.dims[:-1])

        # Pointwise multiplication by complex matrix 
        out_ft = torch.zeros(z0_path.size(), device=z0_path.device, dtype=torch.cfloat)
        if len(self.modes)==2: # 1d case
            out_ft[:, :, freqs[0][0]:freqs[0][1], : ] = compl_mul1d_time(z_ft[:, :, freqs[0][0]:freqs[0][1] ], weights)
        else: # 2d case
            out_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1], : ] = compl_mul2d_time(z_ft[:, :, freqs[0][0]:freqs[0][1], freqs[1][0]:freqs[1][1] ], weights)


        # Compute Inverse FFT
        out_ft = torch.fft.ifftshift(out_ft, dim=self.dims[:-1])
        z = torch.fft.ifftn(out_ft, dim=self.dims[:-1])

        return z.real


#=============================================================================================
# SPDE solver: neural fixed point problem solved by Picard's iteration.
#=============================================================================================

class NeuralFixedPoint(nn.Module):
    def __init__(self, spde_func, n_iter, modes1, modes2, modes3=None):
        super(NeuralFixedPoint, self).__init__()

        # self.padding = int(2**(np.ceil(np.log2(abs(2*T-1)))))

        # number of Picard's iterations
        self.n_iter = n_iter
        
        # vector fields F and G
        self.spde_func = spde_func
        
        # semigroup
        self.convolution = KernelConvolution(spde_func.hidden_channels, modes1, modes2, modes3) 


    def forward(self, z0, xi):
        """ - z0: (batch, hidden_channels, dim_x (possibly dim_y))
            - xi: (batch, forcing_channels, dim_x, (possibly dim_y), dim_t)
        """
        
        # if True 1d, else 2d
        assert len(xi.size()) in [4,5], '1d and 2d cases only are implemented '
        dim_flag = len(xi.size())==4

        # constant path
        if dim_flag:
            z0_path = z0.unsqueeze(-1).repeat(1, 1, 1, xi.size(-1)) 
        else:
            z0_path = z0.unsqueeze(-1).repeat(1, 1, 1, 1, xi.size(-1)) 

        # S_t * z_0
        z0_path =  self.convolution(z0_path, init=True) 

        # step 1 of Picard
        z = z0_path

        # Picard's iterations
        for i in range(self.n_iter):

            F_z, G_z = self.spde_func(z) 

            if dim_flag:
                G_z_xi = torch.einsum('abcde, acde -> abde', G_z, xi)
            else:
                G_z_xi = torch.einsum('abcdef, acdef -> abdef', G_z, xi)

            H_z_xi = F_z + G_z_xi

            y = z0_path + self.convolution(H_z_xi)
            
            z = y
        
        return y