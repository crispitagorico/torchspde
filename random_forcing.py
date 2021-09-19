#TODO: make it compatible with torch version
import torch
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from timeit import default_timer


class WienerInc(object):

    def __init__(self, a=1, J=256, alpha=0.05, device=None):

        self.device = device
        self.J = J 
        self.alpha = alpha 
        self.a = a 

        k_max = J//2

        wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(J,1)

        k_x = wavenumers.transpose(0,1)
        k_y = wavenumers

        lambdax = k_x * 2 * np.pi / a
        lambday = k_y * 2 * np.pi / a


        self.sqrt_eig = torch.exp(- alpha * (lambdax ** 2 + lambday ** 2) / 2)
        self.sqrt_eig[0,0] = 0.0


    def sampledW(self, N, dt, iFspace=False):

        coeff = self.sqrt_eig * np.sqrt(dt) * self.J * self.J / np.sqrt(self.a*self.a)
        
        nn = torch.randn(N, self.J, self.J, 2, device=self.device)

        fft_coeff_real = coeff[None,:,:]*nn[...,0]
        fft_coeff_imag = coeff[None,:,:]*nn[...,1]

        fft_coeff = torch.stack([fft_coeff_real,fft_coeff_imag],dim=-1)

        if iFspace:
            return fft_coeff
        else:

            dW_2copies = torch.fft.ifftn(torch.view_as_complex(fft_coeff), dim=[1,2])
            dW = dW_2copies.real  # only need one copy of dW

            return dW

    def sampleseriessdW(self, N, T, dt, iFspace=False):

        coeff = self.sqrt_eig * np.sqrt(dt) * self.J * self.J / np.sqrt(self.a*self.a)
        
        nn = torch.randn(N, self.J, self.J, T, 2, device=self.device)

        fft_coeff_real = coeff[None,:,:,None]*nn[...,0]
        fft_coeff_imag = coeff[None,:,:,None]*nn[...,1]

        fft_coeff = torch.stack([fft_coeff_real,fft_coeff_imag],dim=-1)

        if iFspace:
            return fft_coeff
        else:

            dW_2copies = torch.fft.ifftn(torch.view_as_complex(fft_coeff), dim=[1,2])
            dW = dW_2copies.real # only need one copy of dW

            return dW



class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, 2, device=self.device)

        coeff[...,0] = self.sqrt_eig*coeff[...,0]
        coeff[...,1] = self.sqrt_eig*coeff[...,1]

        u = torch.fft.ifftn(torch.view_as_complex(coeff), dim=[1,2]).real

        return u