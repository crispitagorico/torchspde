#TODO: Some of the functions below are dupplicated in Wiener.py. Merge the two
import torch
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from timeit import default_timer

def get_twod_bj(dtref,J,a,alpha,device):
    """
    Alg 4.5 Page 443 in the book "An Introduction to Computational Stochastic PDEs"
    """
    lambdax = 2 * np.pi * torch.cat([torch.arange(0,J[0]//2 +1,device=device), torch.arange(- J[0]//2 + 1,0,device=device)]) / a[0]
    lambday = 2 * np.pi * torch.cat([torch.arange(0,J[1]//2 +1,device=device), torch.arange(- J[1]//2 + 1,0,device=device)]) / a[1]
    lambdaxx, lambdayy = torch.meshgrid(lambdax,lambday)
    root_qj = torch.exp(- alpha * (lambdaxx ** 2 + lambdayy ** 2) / 2)
    bj = root_qj * np.sqrt(dtref) * J[0] * J[1] / np.sqrt(a[0] * a[1])
    return bj

def get_twod_dW(bj,kappa,M,device):
    """
    Alg 10.6 Page 444 in the book "An Introduction to Computational Stochastic PDEs"
    """
    J = bj.shape
    if (kappa == 1):
        nn = torch.randn(M,J[0],J[1],2,device=device)
    else:
        nn = torch.sum(torch.randn(kappa,M,J[0],J[1],2,device=device),0)
    nn2 = torch.view_as_complex(nn)
    tmp = torch.fft.ifft2(bj*nn2,dim=[-2,-1])
    dW1 = torch.real(tmp)
    dW2 = torch.imag(tmp)
    return dW1,dW2

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