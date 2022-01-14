# adapted from https://github.com/patrick-kidger/NeuralCDE

import torch
import numpy as np
import torchcde


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        """ TODO: add possibility to have more layers 
        """

        model = [torch.nn.Conv2d(in_size, out_size, 1), torch.nn.BatchNorm2d(out_size), torch.nn.Tanh()] 

        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y)"""
        return self._model(x)

######################
# A CDE model in infinite dimension looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s + \int_0^t g_\theta(z_s)ds
#
# Where z_s is a function of 2 independent space variables
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
        self._F = FNO(modes1=16, modes2=8, in_channels=hidden_size, hidden_channels=hidden_size, L=1) 
        # self._F = MLP(hidden_size, hidden_size)  
        self._G = MLP(hidden_size, hidden_size * noise_size)

    def forward(self, t, z):
        """ z: (batch, hidden_size, dim_x, dim_y)"""

        return self._F(z), self._G(z).view(z.size(0), self._hidden_size, self._noise_size, z.size(2), z.size(3))

    def prod(self, t, z, control_gradient):
        # z is of shape (N, dim_x, dim_y, hidden_channels) 
        # control_gradient is of shape (N, dim_x, dim_y, noise_size)
        
        z = z.permute(0, 3, 1, 2)
        control_gradient = control_gradient.permute(0, 3, 1, 2)

        # z is of shape (N, hidden_channels, dim_x, dim_y) 
        # control_gradient is of shape (N, noise_size, dim_x, dim_y)


        Fu, Gu = self.forward(t, z)
        # Gu is of shape (N, hidden_size, noise_size, dim_x, dim_y)
        # Fu is of shape (N, hidden_size, dim_x, dim_y)

        Guxi = torch.einsum('abcde, acde -> abde', Gu, control_gradient)
        sol = Fu + Guxi
       
        # sol is of shape (N, hidden_size, dim_x, dim_y)
       
        return sol.permute(0, 2, 3, 1)


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
        
        # u0 is of shape #(N, data_size, dim_x, dim_y)
        z0 = self.initial(u0.permute(0, 2, 3, 1))
        # z0 is of shape #(N, dim_x, dim_y, hidden_size)
        # coeffs is of shape #(N, dim_x, dim_y, dim_t, noise_size)

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
        
        # z_T is of shape (N, dim_x, dim_y, dim_t, hidden_channels)
        pred_y = self._readout(z_T).permute(0, 4, 3, 1, 2)
        # pred_y is of shape (N, hidden_channels, dim_t, dim_x, dim_y)
        return pred_y


######################
# A CDE model in infinite dimension looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s + \int_0^t g_\theta(z_s)ds
#
# Here we provide functionalities to model a non local g_\theta 
######################

class FNO(torch.nn.Module):
    def __init__(self, modes1, modes2, in_channels, hidden_channels, L):
        super(FNO, self).__init__()

        """ an FNO to model F(u(t)) where u(t) is a function of 2 spatial variables """

        self.fc0 = torch.nn.Linear(in_channels, hidden_channels)

        self.net = [ FNO_layer(modes1, modes2, hidden_channels) for i in range(L-1) ]
        self.net += [ FNO_layer(modes1, modes2, hidden_channels, last=True) ]
        self.net = torch.nn.Sequential(*self.net)

        self.fc1 = torch.nn.Linear(hidden_channels, 128)
        self.fc2 = torch.nn.Linear(128, in_channels)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y)"""
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = self.net(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.nn.functional.tanh(x)
        return x

class FNO_layer(torch.nn.Module):
    def __init__(self, modes1, modes2, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last
        self.conv = ConvolutionSpace(width, modes1, modes2)
        self.w = torch.nn.Linear(width, width)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y)"""
        x1 = self.conv(x)
        x2 = self.w(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x1 + x2
        if not self.last:
            x = torch.nn.functional.gelu(x)           
        return x


class ConvolutionSpace(torch.nn.Module):
    def __init__(self, channels, modes1, modes2):
        super(ConvolutionSpace, self).__init__()

        """ ...    
        """
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1. / (channels**2)
        self.weights = torch.nn.Parameter(self.scale * torch.rand(channels, channels, self.modes1, self.modes2,  2))


    def forward(self, x):
        """ x: (batch, channels, dim_x, dim_y)"""

        x0, x1 = x.size(2)//2 - self.modes1//2, x.size(2)//2 + self.modes1//2
        y0, y1 = x.size(3)//2 - self.modes2//2, x.size(3)//2 + self.modes2//2

        # Compute FFT of the input signal to convolve
        x_ft = torch.fft.fftn(x, dim=[2, 3])
        x_ft = torch.fft.fftshift(x_ft, dim=[2, 3])

        # Pointwise multiplication by complex matrix 
        out_ft = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), device=x.device, dtype=torch.cfloat)
        out_ft[:, :, x0:x1, y0:y1] = compl_mul2d_spatial(x_ft[:, :, x0:x1, y0:y1], torch.view_as_complex(self.weights))

        # Compute Inverse FFT
        out_ft = torch.fft.ifftshift(out_ft, dim=[2, 3])
        x = torch.fft.ifftn(out_ft, dim=[2, 3], s=(x.size(2), x.size(3)))

        return x.real


def compl_mul2d_spatial(a, b):
    """ ...
    """
    return torch.einsum("aibc, ijbc -> ajbc", a, b)



#===========================================================================
# Data Loaders
#===========================================================================

def dataloader_ncdeinf_2d(u, xi, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=1, batch_size=20, interpolation='linear', dataset=None):

    if dataset=='sns':
        T, sub_t, sub_x = 100, 1, 4
    dim_x = u.size(1)//sub_x

    u_train = u[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t].permute(0, 3, 1, 2).unsqueeze(1) #(N, 1, dim_t, dim_x, dim_y)
    xi_train = xi[:ntrain, ::sub_x, ::sub_x, 0:T].permute(0, 3, 1, 2).unsqueeze(1) #(N, 1, dim_t, dim_x, dim_y)

    u_test = u[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t].permute(0, 3, 1, 2).unsqueeze(1) #(N, 1, dim_t, dim_x, dim_y)
    xi_test = xi[-ntest:, ::sub_x, ::sub_x, 0:T].permute(0, 3, 1, 2).unsqueeze(1) #(N, 1, dim_t, dim_x, dim_y)

    u0_train = u_train[:, :, 0, :, :] #(N, 1, dim_x, dim_y)  
    u0_test = u_test[:, :, 0, :, :]  #(N, 1, dim_x, dim_y)

    # interpolation 
    xi_train = torchcde.linear_interpolation_coeffs(xi_train)
    xi_test = torchcde.linear_interpolation_coeffs(xi_test)

    # interpolation 
    if interpolation == 'linear':
        xi_train = torchcde.linear_interpolation_coeffs(xi_train)
        xi_test = torchcde.linear_interpolation_coeffs(xi_test)
    elif interpolation == 'cubic':
        xi_train = torchcde.hermite_cubic_coefficients_with_backward_differences(xi_train)
        xi_test = torchcde.hermite_cubic_coefficients_with_backward_differences(xi_test)

    xi_train = xi_train.permute(0, 3, 4, 2, 1)  #(N, dim_x, dim_t, 1)
    xi_test = xi_test.permute(0, 3, 4, 2, 1)  #(N, dim_x, dim_t, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


#===========================================================================
# Training functionalities
#===========================================================================

def train_ncdeinf_2d(model, train_loader, test_loader, device, myloss, batch_size=20, epochs=5000, learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    losses_train = []
    losses_test = []

    try:
            
        for ep in range(epochs):

            model.train()
            
            train_loss = 0.
            for u0_, xi_, u_ in train_loader:

                loss = 0.
                
                u0_ = u0_.to(device)
                xi_ = xi_.to(device)
                u_ = u_.to(device)

                u_pred = model(u0_, xi_)
                
                loss = myloss(u_pred[:, :, 1:, :, :].reshape(batch_size, -1), u_[:, :, 1:, :, :].reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            test_loss = 0.
            with torch.no_grad():
                for u0_, xi_, u_ in test_loader:
                    
                    loss = 0.
                    
                    u0_ = u0_.to(device)
                    xi_ = xi_.to(device)
                    u_ = u_.to(device)

                    u_pred = model(u0_, xi_)

                    loss = myloss(u_pred[:, :, 1:, :, :].reshape(batch_size, -1), u_[:, :, 1:, :, :].reshape(batch_size, -1))
                    test_loss += loss.item()

            scheduler.step()
            if ep % print_every == 0:
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Test Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        return model, losses_train, losses_test
    
    except KeyboardInterrupt:

        return model, losses_train, losses_test

