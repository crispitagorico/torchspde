#TODO: dtref and kappa
import torch 
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from time import time

class CylindricalWiener(object):
    
    def __init__(self, s, t, dt, a, b, dx, correlation = None):
        self.s = s 
        self.t = t 
        self.dt = dt 
        self.a = a
        self.b = b 
        self.dx = dx 
        self.correlation = correlation
        # If correlation function is not given, use space-time white noise correlation fucntion
        if self.correlation is None:
            self.correlation = self.WN_corr
        

    def partition(self, a,b, dx): #makes a partition of [a,b] of equal sizes dx
        return np.linspace(a, b, int((b - a) / dx) + 1)
    
    # Create l dimensional Brownian motion with time step = dt
    
    def BM(self, start, stop, dt, l):
        T = self.partition(start, stop, dt)
        # assign to each point of len(T) time point an N(0, \sqrt(dt)) standard l dimensional random variable
        BM = np.random.normal(scale=np.sqrt(dt), size=(len(T), l))
        BM[0] = 0 #set the initial value to 0
        BM = np.cumsum(BM, axis  = 0) # cumulative sum: B_n = \sum_1^n N(0, \sqrt(dt))
        return BM

    # Create space time noise. White in time and with some correlation in space.
    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    # X here is points in space as in SPDE1 function.
    def WN_space_time_single(self):
        
        T, X = self.partition(self.s,self.t, self.dt), self.partition(self.a, self.b, self.dx) #time points, space points,
        N = len(X)
        # Create correlation Matrix in space
        space_corr = np.array([[self.correlation(x, j, self.dx * (N-1)) for j in range(N)] for x in X])
        B = self.BM(self.s, self.t, self.dt, N)
        
        return np.dot(B, space_corr.T)

    def sample(self, num, transpose=True, torch_device=None):
        samples = np.array([self.WN_space_time_single() for _ in range(num)])
        if transpose:
            samples = samples.transpose(0,2,1)[:,:-1,:]
        if torch_device is not None:
            samples = torch.from_numpy(samples.astype(np.float32)).to(torch_device)
        return samples
    
    # Funciton for creating N random initial conditions of the form 
    # \sum_{i = -p}^{i = p} a_k sin(k*\pi*x/scale)/ (1+|k|^decay) where a_k are i.i.d standard normal.
    def initial(self, N, X, p = 10, decay = 2, scaling = 1):
        scale = max(X)/scaling
        IC = []
        SIN = np.array([[np.sin(k*np.pi*x/scale)/((np.abs(k)+1)**decay) for k in range(-p,p+1)] for x in X])
        for i in range(N):
            sins = np.random.normal(size = 2*p+1)
            extra = np.random.normal(size = 1)
            IC.append(np.dot(SIN, sins)+extra)
            # enforcing periodic boundary condition without error
            IC[-1][-1] = IC[-1][0]
        
        return np.array(IC) 
    
    #Correlation function that approximates WN in space.
    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    def WN_corr(self, x, j, a):
        return np.sqrt(2 / a) * np.sin(j * np.pi * x / a)


class QWiener2D(object):

    def __init__(self, dtref, kappa, a, J, alpha):
        self.a = a 
        self.J = J
        self.dtref = dtref
        self.kappa = kappa
        self.alpha = alpha

    def get_twod_bj(self, device):
        """
        Alg 4.5 Page 443 in the book "An Introduction to Computational Stochastic PDEs"
        """
        lambdax = 2 * np.pi * torch.cat([torch.arange(0,self.J[0]//2 +1,device=device), torch.arange(- self.J[0]//2 + 1,0,device=device)]) / self.a[0]
        lambday = 2 * np.pi * torch.cat([torch.arange(0,self.J[1]//2 +1,device=device), torch.arange(- self.J[1]//2 + 1,0,device=device)]) / self.a[1]
        lambdaxx, lambdayy = torch.meshgrid(lambdax,lambday)
        root_qj = torch.exp(- self.alpha * (lambdaxx ** 2 + lambdayy ** 2) / 2)
        bj = root_qj * np.sqrt(self.dtref) * self.J[0] * self.J[1] / np.sqrt(self.a[0] * self.a[1])
        return bj

    def get_twod_dW(self,bj,M,device):
        """
        Alg 10.6 Page 444 in the book "An Introduction to Computational Stochastic PDEs"
        """
        J = bj.shape
        if (self.kappa == 1):
            nn = torch.randn(M,self.J[0],self.J[1],2,device=device)
        else:
            nn = torch.sum(torch.randn(self.kappa,M,self.J[0],self.J[1],2,device=device),0)
        nn2 = torch.view_as_complex(nn)
        tmp = torch.fft.ifft2(bj*nn2,dim=[-2,-1])
        dW1 = torch.real(tmp)
        dW2 = torch.imag(tmp)
        return dW1

    def sample(self, num, torch_device=None):
        bj = self.get_twod_bj(torch_device)
        dW = self.get_twod_dW(bj,num,torch_device)
        return dW



