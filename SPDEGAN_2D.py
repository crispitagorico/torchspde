# adapted from https://github.com/google-research/torchsde/blob/master/examples/sde_gan.py

import torch
import numpy as np
import torch.nn.functional as F
from SPDE2Dint import NeuralFixedPoint
from fourier_space2d_time import FNO_layer 

#####
# TODO: change F into a FNO, discriminator. 
####

###################
# First some standard helper objects.
###################
class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

###################
# MLP layers are used to 
# model F and G (local operators) in the SPDE 
###################
class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Conv3d(in_size, mlp_size, 1),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Conv3d(mlp_size, mlp_size, 1))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(torch.nn.Conv3d(mlp_size, out_size, 1))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        # x: (batch, in_size, dim_x, dim_y, dim_t)
        return self._model(x)


class MLP2D(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Conv2d(in_size, mlp_size, 1),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Conv2d(mlp_size, mlp_size, 1))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(torch.nn.Conv2d(mlp_size, out_size, 1))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        # x: (batch, in_size, dim_x, dim_y, dim_t)
        return self._model(x)

###################
# Now we define the SPDEs.
#
# We begin by defining the generator SPDE.
###################
class GeneratorFunc(torch.nn.Module):
    # SPDE in functional form: partial_t u_t = F(u_t) + G(u_t) partial_t xi(t) 

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size
        
        ###################
        # F and G are resolution invariant MLP (acting on the channels). 
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        ###################
        self._F = MLP(hidden_size, hidden_size, mlp_size, num_layers, tanh=True)  # add dimensions to hidden_size if grid is used in input
        self._G = MLP(hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)

    def forward(self, u):
        # t has shape ()
        # u has shape (batch_size, hidden_size, dim_x, dim_y, dim_t)

        # if we want to add the space-time grid
        # t = t.expand(x.size(0), 1)
        # tx = torch.cat([t, x], dim=1)
        return self._F(u), self._G(u).view(u.size(0), self._hidden_size, self._noise_size, u.size(2), u.size(3), u.size(4))


###################
# Now we wrap it up into something that computes the SPDE.
###################
class Generator(torch.nn.Module):
    def __init__(self, ic, wiener, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers, modes1, modes2, modes3, T):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self._wiener = wiener
        self._ic = ic

        self._initial = MLP2D(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Conv3d(hidden_size, data_size, 1)

        self._integrator = NeuralFixedPoint(self._func, modes1, modes2, modes3, T, n_iter=2)

    def forward(self, device, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SPDE at.

        ###################
        # Actually solve the SPDE. 
        ###################
        # init_noise = torch.randn(batch_size, self._initial_noise_size, device=device)  
        init_noise = self._ic.sample(batch_size, device=device)
        z0 = self._initial(init_noise)  

        ###################
        # Sample Q-Wiener process  
        ###################
        xi = self._wiener.sample(batch_size, torch_device = device)
        xi = xi.unsqueeze(1)

        ################### 
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method. 
        ###################
        # xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1.0,     
                                    #  adjoint_method='adjoint_reversible_heun',)

        zs = self._integrator(z0, xi)

        ys = self._readout(zs)

        ###################
        # Normalise the data to the form that the discriminator expects
        ###################
        
        return ys

###################
# Next the discriminator. Here, we're going to use our model for SPDE as the
# discriminator. Except that the forcing will not be random but the output of the generator instead.
# TODO: input the derivative instead. 
###################

class DiscriminatorSPDE(torch.nn.Module): 
    def __init__(self, data_size, hidden_size, mlp_size, num_layers, modes1, modes2, modes3, T):
        super().__init__()

        self._readin = MLP(data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._net =  NeuralFixedPoint(self._func, modes1, modes2, modes3, T, n_iter=2)
        # self._net = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        
        self._readout = torch.nn.Conv2d(hidden_size, 1, 1)


    def forward(self, x):
        """ - x: (batch, data_size, dim_x, dim_t)
        """

        x = self._readin(x)

        x = self._net(x)

        x = self._readout(x)  # (batch, 1, dim_x, dim_y, dim_t)

        x = x[...,-1]          # (batch, 1, dim_x, dim_y)
 
        score = x[...,0,-1,-1]    # (batch)

        return score.mean()

###################
# Next the discriminator. Here, we're going to use a 2D fourier neural operator (FNO) as the
# discriminator. 
###################

class DiscriminatorFNO(torch.nn.Module): 
    def __init__(self, data_size, hidden_size, mlp_size, num_layers, modes1, modes2, modes3):
        super().__init__()

        self._readin = MLP(data_size+3, hidden_size, mlp_size, num_layers, tanh=False)
        self._net =  [ FNO_layer(modes1, modes2, modes3, hidden_size) for i in range(num_layers-1) ]
        self._net += [ FNO_layer(modes1, modes2, modes3, hidden_size, last=True) ]
        self._net = torch.nn.Sequential(*self._net)
        self._readout = torch.nn.Conv3d(hidden_size, 1, 1)

    def forward(self, x):
        """ - x: (batch, data_size, dim_x, dim_y, dim_t)
        """
        grid = self.get_grid(x[:,0,...], x.device)
        x = torch.cat((x, grid), dim=1)

        x = self._readin(x)

        x = self._net(x)

        x = self._readout(x)  # (batch, 1, dim_x, dim_y, dim_t)
   
        x = x[...,-1]         # (batch, 1, dim_x, dim_y)

        score = x[...,0,-1,-1]  # (batch)

        return score.mean()

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).permute(0,4,1,2,3).to(device)

