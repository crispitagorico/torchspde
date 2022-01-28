import torch
import csv
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .utils import UnitGaussianNormalizer
from utilities import LpLoss, count_params, EarlyStopping

class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class ConvNet(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        if dim==2:
            self.conv1 = nn.Conv2d(1, size[0], 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(size[0], size[1], 5)
        elif dim==3:
            self.conv1 = nn.Conv3d(1, size[0], 5)
            self.pool = nn.MaxPool3d(2, 2)
            self.conv2 = nn.Conv3d(size[0], size[1], 5)
        self.fc1 = nn.Linear(size[1] * 5 * 5 * 5, size[2])
        self.fc2 = nn.Linear(size[2], size[3])
        self.fc3 = nn.Linear(size[3], size[4])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepONetCP(nn.Module):
    def __init__(self, branch_layer, trunk_layer, conv=0):
        super(DeepONetCP, self).__init__()
        if conv>0:
            self.branch = ConvNet(branch_layer, conv)
        else:
            self.branch = DenseNet(branch_layer, nn.ReLU, nn.Tanh)
        self.trunk = DenseNet(trunk_layer, nn.ReLU, nn.Tanh)

    def forward(self, u0, grid):
        a = self.branch(u0)
        # batchsize x width
        b = self.trunk(grid)
        # N x width
        return torch.einsum('bi,ni->bn', a, b)


#===========================================================================
# Data Loaders for Neural SPDE
#===========================================================================

def dataloader_deeponet_1d_xi(u, xi, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, normalizer=False, dataset=None):

    if dataset=='phi41':
        T, sub_t, dim_t = 51, 1, 50
    elif dataset=='wave':
        T, sub_t = (u.shape[-1]+1)//2, 5  #TODO: dim_t

    u_train = u[:ntrain, :dim_x, 1:T:sub_t].reshape(ntrain, -1)
    xi_train = torch.diff(xi[:ntrain, :-1, 0:T:sub_t], dim=-1)
    dim_t = xi_train.shape[-1]
    xi_train = xi_train.reshape(ntrain, -1)

    u_test = u[-ntest:, :dim_x, 1:T:sub_t].reshape(ntest, -1)
    xi_test = torch.diff(xi[-ntest:, :-1, 0:T:sub_t],dim=-1)
    xi_test = xi_test.reshape(ntest, -1)

    if normalizer:
        xi_normalizer = UnitGaussianNormalizer(xi_train)
        xi_train = xi_normalizer.encode(xi_train)
        xi_test = xi_normalizer.encode(xi_test)

        normalizer = UnitGaussianNormalizer(u_train)
        u_train = normalizer.encode(u_train)
        u_test = normalizer.encode(u_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_test, u_test), batch_size=batch_size, shuffle=False)

    gridx = torch.tensor(np.linspace(0, 1, dim_x), dtype=torch.float)
    gridx = gridx.reshape(1, dim_x, 1, 1).repeat([batch_size, 1, dim_t, 1])
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float)
    gridt = gridt.reshape(1, 1, dim_t, 1).repeat([batch_size, dim_x, 1, 1])
    grid =  torch.cat((gridx, gridt), dim=-1)[0].reshape(dim_x * dim_t, 2)

    return train_loader, test_loader, normalizer, grid

# def dataloader_deeponet_conv_1d_xi(u, xi, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, normalizer=False, dataset=None):

#     if dataset=='phi41':
#         T, sub_t, dim_t = 51, 1, 50
#     elif dataset=='wave':
#         T, sub_t = (u.shape[-1]+1)//2, 5  #TODO: dim_t

#     u_train = u[:ntrain, :dim_x, 1:T:sub_t]
#     xi_train = torch.diff(xi[:ntrain, :-1, 0:T:sub_t], dim=-1)
#     dim_t = xi_train.shape[-1]

#     u_test = u[-ntest:, :dim_x, 1:T:sub_t]
#     xi_test = torch.diff(xi[-ntest:, :-1, 0:T:sub_t],dim=-1)

#     if normalizer:
#         xi_normalizer = UnitGaussianNormalizer(xi_train)
#         xi_train = xi_normalizer.encode(xi_train)
#         xi_test = xi_normalizer.encode(xi_test)

#         normalizer = UnitGaussianNormalizer(u_train)
#         u_train = normalizer.encode(u_train)
#         u_test = normalizer.encode(u_test)

#     train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_train, u_train), batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_test, u_test), batch_size=batch_size, shuffle=False)

#     gridx = torch.tensor(np.linspace(0, 1, dim_x), dtype=torch.float)
#     gridx = gridx.reshape(1, dim_x, 1, 1).repeat([batch_size, 1, dim_t, 1])
#     gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float)
#     gridt = gridt.reshape(1, 1, dim_t, 1).repeat([batch_size, dim_x, 1, 1])
#     grid =  torch.cat((gridx, gridt), dim=-1)[0].reshape(dim_x * dim_t, 2)

#     return train_loader, test_loader, normalizer, grid

def dataloader_deeponet_1d_u0(u, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, normalizer=False, dataset=None):

    if dataset=='phi41':
        T, sub_t, dim_t = 51, 1, 50
    elif dataset=='wave':
        T, sub_t = (u.shape[-1]+1)//2, 5

    u0_train = u[:ntrain, :dim_x, 0].reshape(ntrain, -1)
    u_train = u[:ntrain, :dim_x, 1:T:sub_t] 
    dim_t = u_train.shape[-1]
    u_train = u_train.reshape(ntrain, -1)

    u0_test = u[-ntest:, :dim_x, 0].reshape(ntest, -1)
    u_test = u[-ntest:, :dim_x, 1:T:sub_t].reshape(ntest, -1)

    if normalizer:
        u_normalizer = UnitGaussianNormalizer(u_train)
        u_train = u_normalizer.encode(u_train)
        u_test = u_normalizer.encode(u_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, u_test), batch_size=batch_size, shuffle=False)

    gridx = torch.tensor(np.linspace(0, 1, dim_x), dtype=torch.float)
    gridx = gridx.reshape(1, dim_x, 1, 1).repeat([batch_size, 1, dim_t, 1])
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float)
    gridt = gridt.reshape(1, 1, dim_t, 1).repeat([batch_size, dim_x, 1, 1])
    grid =  torch.cat((gridx, gridt), dim=-1)[0].reshape(dim_x * dim_t, 2)

    return train_loader, test_loader, u_normalizer, grid


def dataloader_deeponet_2d_xi(u, xi, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=4, batch_size=20, normalizer=128, dataset=None, conv=False):

    if dataset=='sns':
        T, sub_t, sub_x = 51, 1, 4

    u_train = u[:ntrain, ::sub_x, ::sub_x, 1:T:sub_t].reshape(ntrain, -1)
    xi_train = xi[:ntrain, ::sub_x, ::sub_x, 1:T:sub_t]
    dim_x = xi_train.shape[1]
    dim_t = xi_train.shape[-1]
    if not conv:
        xi_train = xi_train.reshape(ntrain, -1)
    
    u_test = u[-ntest:, ::sub_x, ::sub_x, 1:T:sub_t].reshape(ntest, -1)
    xi_test = xi[-ntest:, ::sub_x, ::sub_x, 1:T:sub_t]

    if not conv:
        xi_test = xi_test.reshape(ntest, -1)

    if normalizer:
        xi_normalizer = UnitGaussianNormalizer(xi_train)
        xi_train = xi_normalizer.encode(xi_train)
        xi_test = xi_normalizer.encode(xi_test)

        normalizer = UnitGaussianNormalizer(u_train)
        u_train = normalizer.encode(u_train)
        u_test = normalizer.encode(u_test)

    gridx = torch.tensor(np.linspace(0, 1, dim_x), dtype=torch.float)
    gridx = gridx.reshape(1, dim_x, 1, 1, 1).repeat([batch_size, 1, dim_x, dim_t, 1])
    gridy = torch.tensor(np.linspace(0, 1, dim_x), dtype=torch.float)
    gridy = gridy.reshape(1, 1, dim_x, 1, 1).repeat([batch_size, dim_x, 1,  dim_t, 1])
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, dim_t, 1).repeat([batch_size, dim_x, dim_x, 1, 1])
    grid =  torch.cat((gridx, gridy, gridt), dim=-1)[0].reshape(dim_x * dim_x * dim_t, 3)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, normalizer, grid

def dataloader_deeponet_2d_u0(u, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=4, batch_size=20, normalizer=128, dataset=None, conv=False):

    if dataset=='sns':
        T, sub_t, sub_x = 51, 1, 4

    u0_train = u[:ntrain, ::sub_x, ::sub_x, 0]

    if not conv:
        u0_train = u0_train.reshape(ntrain, -1)

    u_train = u[:ntrain, ::sub_x, ::sub_x, 1:T:sub_t]
    dim_t = u_train.shape[-1]
    dim_x = u_train.shape[1]
    u_train = u_train.reshape(ntrain, -1)

    u0_test = u[-ntest:, ::sub_x, ::sub_x, 0] 
    if not conv:
        u0_test = u0_test.reshape(ntest, -1)
    u_test = u[-ntest:, ::sub_x, ::sub_x, 1:T:sub_t].reshape(ntest, -1)

    if normalizer:
        u0_normalizer = UnitGaussianNormalizer(u0_train)
        u0_train = u0_normalizer.encode(u0_train)
        u0_test = u0_normalizer.encode(u0_test)

        normalizer = UnitGaussianNormalizer(u_train)
        u_train = normalizer.encode(u_train)
        u_test = normalizer.encode(u_test)

    gridx = torch.tensor(np.linspace(0, 1, dim_x), dtype=torch.float)
    gridx = gridx.reshape(1, dim_x, 1, 1, 1).repeat([batch_size, 1, dim_x, dim_t, 1])
    gridy = torch.tensor(np.linspace(0, 1, dim_x), dtype=torch.float)
    gridy = gridy.reshape(1, 1, dim_x, 1, 1).repeat([batch_size, dim_x, 1,  dim_t, 1])
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, dim_t, 1).repeat([batch_size, dim_x, dim_x, 1, 1])
    grid =  torch.cat((gridx, gridy, gridt), dim=-1)[0].reshape(dim_x * dim_x * dim_t, 3)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, normalizer, grid


#===========================================================================
# Training and Testing functionalities
#===========================================================================
def eval_deeponet(model, test_dl, myloss, batch_size, device, grid, u_normalizer=None):

    grid = grid.to(device)
    ntest = len(test_dl.dataset)
    test_loss = 0.
    with torch.no_grad():
        for u0_, u_ in test_dl:    
            loss = 0.       
            u0_,  u_ = u0_.to(device),  u_.to(device)
            u_pred = model(u0_, grid)

            if u_normalizer is not None:
                u_pred = u_normalizer.decode(u_pred.cpu())
                u_ = u_normalizer.decode(u_.cpu())

            loss = myloss(u_pred.reshape(batch_size, -1), u_.reshape(batch_size, -1))
            test_loss += loss.item()
    print('Test Loss: {:.6f}'.format(test_loss / ntest))
    return test_loss / ntest

def train_deepOnet_1d(model, train_loader, test_loader, grid, u_normalizer, device, myloss, batch_size=20, epochs=5000, learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20, plateau_patience=None, plateau_terminate=None, checkpoint_file='checkpoint.pt'):
    
    grid = grid.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    if plateau_patience is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=plateau_patience, threshold=1e-6, min_lr=1e-7)
    if plateau_terminate is not None:
        early_stopping = EarlyStopping(patience=plateau_terminate, verbose=False, path=checkpoint_file)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)
    
    losses_train = []
    losses_test = []

    try:
        for ep in range(epochs):

            model.train()
            
            train_loss = 0.
            for u0_ , u_ in train_loader:
                
                loss = 0.

                u0_ = u0_.to(device)
                u_ = u_.to(device)
                
                u_pred = model(u0_, grid)

                if u_normalizer is not None:
                    u_pred = u_normalizer.decode(u_pred.cpu())
                    u_ = u_normalizer.decode(u_.cpu())
                
                loss = myloss(u_pred.reshape(batch_size, -1), u_.reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            test_loss = 0.
            with torch.no_grad():
                for u0_, u_ in test_loader:
                    
                    loss = 0.
                    
                    u0_ = u0_.to(device)
                    u_ = u_.to(device)

                    u_pred = model(u0_, grid)
                    
                    if u_normalizer is not None:
                        u_pred = u_normalizer.decode(u_pred.cpu())
                        u_ = u_normalizer.decode(u_.cpu())

                    loss = myloss(u_pred.reshape(batch_size, -1), u_.reshape(batch_size, -1))

                    test_loss += loss.item()
            
            if plateau_patience is None:
                scheduler.step()
            else:
                scheduler.step(test_loss/ntest)
            if plateau_terminate is not None:
                early_stopping(test_loss/ntest, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if ep % print_every == 0:
        
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Test Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        return model, losses_train, losses_test
    
    except KeyboardInterrupt:

        return model, losses_train, losses_test


def hyperparameter_search(train_dl, val_dl, test_dl, S, grid, u_normalizer=None, width=[128,256,512], branch_depth=[2,3,4], trunk_depth=[2,3,4], epochs=500, print_every=20, lr=0.025, plateau_patience=100, plateau_terminate=100, log_file ='log_nspde', checkpoint_file='checkpoint.pt', final_checkpoint_file='final.pt'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparams = list(itertools.product(width, branch_depth, trunk_depth ))

    loss = LpLoss(size_average=False)
    
    fieldnames = ['width','bd','td', 'nb_params', 'loss_train', 'loss_val', 'loss_test']
    with open(log_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        
    best_loss_val = 1000.

    for (w, bd, td) in hyperparams:
        
        print('\n width:{}, branch depth:{},  trunk depth:{}'.format(w, bd, td))

        branch = [S]+ bd*[w] 
        trunk = [grid.shape[-1]] + td*[w]

        model = DeepONetCP(branch_layer=branch,
                    trunk_layer=trunk).to(device)
        nb_params = count_params(model)
        
        print('\n The model has {} parameters'. format(nb_params))

        # Train the model. The best model is checkpointed.
        _, _, _ = train_deepOnet_1d(model, train_dl, val_dl, grid, u_normalizer, device, loss, batch_size=20, epochs=epochs, learning_rate=lr, scheduler_step=500, scheduler_gamma=0.5, print_every=print_every, plateau_patience=plateau_patience, plateau_terminate=plateau_terminate, checkpoint_file=checkpoint_file)

        # load the best trained model 
        model.load_state_dict(torch.load(checkpoint_file))
        
        # compute the test loss 
        loss_test = eval_deeponet(model, test_dl, loss, 20, device, grid, u_normalizer=u_normalizer)
        # we also recompute the validation and train loss
        loss_train = eval_deeponet(model, train_dl, loss, 20, device, grid, u_normalizer=u_normalizer)
        loss_val = eval_deeponet(model, val_dl, loss, 20, device, grid, u_normalizer=u_normalizer)

        # if this configuration of hyperparameters is the best so far (determined wihtout using the test set), save it 
        if loss_val < best_loss_val:
            torch.save(model.state_dict(), final_checkpoint_file)
            best_loss_val = loss_val

        # write results
        with open(log_file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([w, bd, td, nb_params, loss_train, loss_val, loss_test])

