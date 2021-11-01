# adapted from https://github.com/patrick-kidger/NeuralCDE

import torch
import numpy as np
import torchcde


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        """ TODO: add possibility to have more layers 
        """

        model = [torch.nn.Conv1d(in_size, out_size, 1), torch.nn.BatchNorm1d(out_size), torch.nn.Tanh()] 

        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x)"""
        return self._model(x)

######################
# A CDE model in infinite dimension looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s + \int_0^t g_\theta(z_s)ds
#
# Where z_s is a function of 1 independent space variables
# and where X is your data and f_\theta and g_\theta are neural networks. 
#
# So the first thing we need to do is define such an f_\theta and a g_\theta
# That's what this CDEFunc class does.
######################


class CDEFunc(torch.nn.Module):

    def __init__(self, noise_size, hidden_size):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        # F and G are resolution invariant MLP (acting on the channels). 
        self._F = FNO(modes=16, in_channels=hidden_size, hidden_channels=hidden_size, L=1) #MLP(hidden_size, hidden_size)  
        self._G = MLP(hidden_size, hidden_size * noise_size)

    def forward(self, t, z):
        """ z: (batch, hidden_size, dim_x)"""

        return self._F(z), self._G(z).view(z.size(0), self._hidden_size, self._noise_size, z.size(2))

    def prod(self, t, z, control_gradient):
        # z is of shape (N, dim_x, hidden_channels) 
        # control_gradient is of shape (N, dim_x, noise_size)
        
        z = z.permute(0,2,1)
        control_gradient = control_gradient.permute(0,2,1)

        # z is of shape (N, hidden_channels, dim_x) 
        # control_gradient is of shape (N, noise_size, dim_x)


        Fu, Gu = self.forward(t, z)
        # Gu is of shape (N, hidden_size, noise_size, dim_x)
        # Fu is of shape (N, hidden_size, dim_x)

        Guxi = torch.einsum('abcd, acd -> abd', Gu, control_gradient)
        sol = Fu + Guxi
       
        # sol is of shape (N, hidden_size, dim_x)
       
        return sol.permute(0,2,1)


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, data_size, noise_size, hidden_channels, output_channels, interpolation="linear"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(noise_size, hidden_channels)
        self.initial = torch.nn.Linear(data_size, hidden_channels)
        
        # self.readout = torch.nn.Linear(hidden_channels, output_channels)
        readout = [torch.nn.Linear(hidden_channels, 128), torch.nn.ReLU(), torch.nn.Linear(128, output_channels)]
        self._readout = torch.nn.Sequential(*readout)
        self.interpolation = interpolation

    def forward(self, u0, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        
        # u0 is of shape #(N, data_size, dim_x)
        z0 = self.initial(u0.permute(0, 2, 1))
        # z0 is of shape #(N, dim_x, hidden_size)
        # coeffs is of shape #(N, dim_x, dim_t, noise_size)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              # t = X.interval,
                              method='euler',
                              t=X._t)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        # z_T = z_T[:, 1]
        
        # z_T is of shape (N, dim_x, dim_t, hidden_channels)
        pred_y = self._readout(z_T).permute(0, 3, 2, 1)
        # pred_y is of shape (N, hidden_channels, dim_t, dim_x)
        return pred_y


######################
# A CDE model in infinite dimension looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s + \int_0^t g_\theta(z_s)ds
#
# Here we provide functionalities to model a non local g_\theta 
######################

class FNO(torch.nn.Module):
    def __init__(self, modes, in_channels, hidden_channels, L):
        super(FNO, self).__init__()

        """ an FNO to model F(u(t)) where u(t) is a function of 1 spatial variable """

        self.fc0 = torch.nn.Linear(in_channels, hidden_channels)

        self.net = [ FNO_layer(modes, hidden_channels) for i in range(L-1) ]
        self.net += [ FNO_layer(modes, hidden_channels, last=True) ]
        self.net = torch.nn.Sequential(*self.net)

        self.fc1 = torch.nn.Linear(hidden_channels, 128)
        self.fc2 = torch.nn.Linear(128, in_channels)

    def forward(self, x):
        """ x: (batch, in_channels, dim_x)"""
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x = self.net(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)

        return x

class FNO_layer(torch.nn.Module):
    def __init__(self, modes, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last
        self.conv = ConvolutionSpace(width, modes)
        self.w = torch.nn.Linear(width, width)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x)"""
        x1 = self.conv(x)
        x2 = self.w(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x1 + x2
        if not self.last:
            x = torch.nn.functional.gelu(x)           
        return x


class ConvolutionSpace(torch.nn.Module):
    def __init__(self, channels, modes):
        super(ConvolutionSpace, self).__init__()

        """ ...    
        """
        self.modes = modes
        self.scale = 1. / (channels**2)
        self.weights = torch.nn.Parameter(self.scale * torch.rand(channels, channels, modes,  2))


    def forward(self, x):
        """ x: (batch, channels, dim_x)"""

        x0, x1 = x.size(2)//2 - self.modes//2, x.size(2)//2 + self.modes//2

        # Compute FFT of the input signal to convolve
        x_ft = torch.fft.fftn(x, dim=[2])
        x_ft = torch.fft.fftshift(x_ft, dim=[2])

        # Pointwise multiplication by complex matrix 
        out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), device=x.device, dtype=torch.cfloat)
        out_ft[:, :, x0:x1] = compl_mul1d_spatial(x_ft[:, :, x0:x1], torch.view_as_complex(self.weights))

        # Compute Inverse FFT
        out_ft = torch.fft.ifftshift(out_ft, dim=[2])
        x = torch.fft.ifftn(out_ft, dim=[2], s=x.size(2))

        return x.real

def compl_mul1d_spatial(a, b):
    """ ...
    """
    return torch.einsum("aib, ijb -> ajb", a, b)

