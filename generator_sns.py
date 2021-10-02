#TODO: make it compatible with torch version
# adapted from https://github.com/zongyi-li/fourier_neural_operator. 
# on top of the deterministic forcing with give the possibility to add a random noise which is a Wiener process 
# in two dimensions following Example 10.3 and 10.12 in the book
# An Introduction to Computational Stochastic PDEs

import torch

import math

import matplotlib.pyplot as plt
import matplotlib
from tqdm.notebook import tqdm

from random_forcing import GaussianRF, get_twod_bj, get_twod_dW

from timeit import default_timer

import scipy.io


#a: domain where we are solving
#w0: initial vorticity
#f: deterministic forcing term 
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def navier_stokes_2d(a, w0, f, visc, T, delta_t=1e-4, record_steps=1, stochastic_forcing=None):

    #Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1] 

    #Maximum frequency
    k_max1 = math.floor(N1/2.0)
    k_max2 = math.floor(N1/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1,2])
    w_h = torch.stack([w_h.real, w_h.imag],dim=-1)

    #Forcing to Fourier space
    if f is not None:
        f_h = torch.fft.fftn(f, dim=[-2,-1])
        f_h = torch.stack([f_h.real, f_h.imag],dim=-1) 
        #If same forcing for the whole batch 
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
    else:
        f_h = torch.zeros_like(w_h)
        
    #If stochastic forcing
    if stochastic_forcing is not None:
        # initialise noise
        bj = get_twod_bj(delta_t,[N1,N2],a,stochastic_forcing['alpha'],w_h.device)

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=w0.device), torch.arange(start=-k_max2, end=0, step=1, device=w0.device)), 0).repeat(N1,1)
    #Wavenumbers in x-direction
    k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=w0.device), torch.arange(start=-k_max1, end=0, step=1, device=w0.device)), 0).repeat(N2,1).transpose(0,1)
    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2/a[0]**2 + k_y**2/a[1]**2)
    # lap_ = lap.clone()
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max2, torch.abs(k_x) <= (2.0/3.0)*k_max1).float(), 0)

    #Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    forcing = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in tqdm(range(steps)):
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[...,0] = psi_h[...,0]/lap
        psi_h[...,1] = psi_h[...,1]/lap

        #Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[...,0].clone()
        q[...,0] = -2*math.pi*k_y*q[...,1]
        q[...,1] = 2*math.pi*k_y*temp
        q = torch.fft.ifftn(torch.view_as_complex(q/a[1]), dim=[1,2], s=(N1,N2)).real

        #Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[...,0].clone()
        v[...,0] = 2*math.pi*k_x*v[...,1]
        v[...,1] = -2*math.pi*k_x*temp
        v = torch.fft.ifftn(torch.view_as_complex(v/a[0]), dim=[1,2], s=(N1,N2)).real

        #Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[...,0].clone()
        w_x[...,0] = -2*math.pi*k_x*w_x[...,1]
        w_x[...,1] = 2*math.pi*k_x*temp
        w_x = torch.fft.ifftn(torch.view_as_complex(w_x/a[0]), dim=[1,2], s=(N1,N2)).real

        #Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[...,0].clone()
        w_y[...,0] = -2*math.pi*k_y*w_y[...,1]
        w_y[...,1] = 2*math.pi*k_y*temp
        w_y = torch.fft.ifftn(torch.view_as_complex(w_y/a[1]), dim=[1,2], s=(N1,N2)).real

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.fftn(q*w_x + v*w_y, dim=[1,2])
        F_h = torch.stack([F_h.real, F_h.imag],dim=-1) 

        #Dealias
        F_h[...,0] = dealias* F_h[...,0]
        F_h[...,1] = dealias* F_h[...,1]

        #Cranck-Nicholson update

        if stochastic_forcing:
          dW, dW2 = get_twod_dW(bj, stochastic_forcing['kappa'], w_h.shape[0], w_h.device)
          gudWh = torch.fft.fft2(stochastic_forcing['sigma']*dW, dim=[-2,-1])
          gudWh = torch.stack([gudWh.real, gudWh.imag],dim=-1) 
        else:
          gudWh = torch.zeros_like(f_h)

        w_h[...,0] = (-delta_t*F_h[...,0] + delta_t*f_h[...,0] + gudWh[...,0] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,0])/(1.0 + 0.5*delta_t*visc*lap)
        w_h[...,1] = (-delta_t*F_h[...,1] + delta_t*f_h[...,1] + gudWh[...,1] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,1])/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1,2], s=(N1,N2)).real
            if stochastic_forcing:
              forcing[...,c] = dW
            #Record solution and time
            sol[...,c] = w
            sol_t[c] = t

            c += 1

    if stochastic_forcing:
      return sol, sol_t, forcing

    return sol, sol_t


# ### Example to generate data

# device = torch.device('cuda')

# # Viscosity parameter
# nu = 1e-4

# # Spatial Resolution
# s = 256
# sub = 1

# # Temporal Resolution   
# T = 25.0
# delta_t = 1e-3
# steps = math.ceil(T/delta_t)

# # Number of solutions to generate
# N = 20

# # Set up 2d GRF with covariance parameters
# GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

# # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
# t = torch.linspace(0, 1, s+1, device=device)
# t = t[0:-1]

# X,Y = torch.meshgrid(t, t)
# f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

# # Stochastic forcing function: sigma*dW/dt 
# stochastic_forcing = {'alpha':0.05, 'kappa':1, 'sigma':0.05}


# # Number of snapshots from solution
# record_steps = 26

# # Inputs
# a = torch.zeros(N, s, s)
# # Solutions
# u = torch.zeros(N, s, s, record_steps)

# # Solve equations in batches (order of magnitude speed-up)

# # Batch size
# bsize = 20

# c = 0
# t0 =default_timer()
# for j in range(N//bsize):

#     #Sample random feilds
#     w0 = GRF.sample(1).repeat(bsize,1,1)#bsize)

#     sol, sol_t, forcing = navier_stokes_2d([1,1], w0, f, nu, T, delta_t, record_steps, stochastic_forcing)  

#     a[c:(c+bsize),...] = w0
#     u[c:(c+bsize),...] = sol

#     c += bsize
#     t1 = default_timer()
#     print(j, c, t1-t0)

# scipy.io.savemat('ns_data.mat', mdict={'t': sol_t, 'sol': sol.cpu().numpy(), 'forcing': forcing.cpu().numpy(), 'param':stochastic_forcing})