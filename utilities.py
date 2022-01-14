import torch
import scipy.io
import h5py
import operator
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import SubplotSpec
from matplotlib.ticker import MaxNLocator
from functools import reduce
from functools import partial 
from timeit import default_timer


#===========================================================================
# Data Loaders for Neural SPDE
#===========================================================================

def dataloader_nspde_1d(u, xi=None, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, dataset=None):

    if xi is None:
        print('There is no known forcing')

    if dataset=='phi41':
        T, sub_t = 51, 1
    elif dataset=='wave':
        T, sub_t = (u.shape[-1]+1)//2, 5

    u0_train = u[:ntrain, :dim_x, 0].unsqueeze(1)
    u_train = u[:ntrain, :dim_x, :T:sub_t]

    if xi is not None:
        xi_train = torch.diff(xi[:ntrain, :dim_x, 0:T:sub_t], dim=-1).unsqueeze(1)
        xi_train = torch.cat([torch.zeros_like(xi_train[..., 0].unsqueeze(-1)), xi_train], dim=-1)
    else:
        xi_train = torch.zeros_like(u_train)

    u0_test = u[-ntest:, :dim_x, 0].unsqueeze(1)
    u_test = u[-ntest:, :dim_x, 0:T:sub_t]

    if xi is not None:
        xi_test = torch.diff(xi[-ntest:, :dim_x, 0:T:sub_t], dim=-1).unsqueeze(1)
        xi_test = torch.cat([torch.zeros_like(xi_test[..., 0].unsqueeze(-1)), xi_test], dim=-1)
    else:
        xi_test = torch.zeros_like(u_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def dataloader_nspde_2d(u, xi=None, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=1, batch_size=20, dataset=None):

    if xi is None:
        print('There is no known forcing')

    if dataset=='sns':
        T, sub_t = 51, 1

    u0_train = u[:ntrain, ::sub_x, ::sub_x, 0].unsqueeze(1)
    u_train = u[:ntrain, ::sub_x, ::sub_x, :T:sub_t]

    if xi is not None:
        xi_train = xi[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t].unsqueeze(1)
    else:
        xi_train = torch.zeros_like(u_train)

    u0_test = u[-ntest:, ::sub_x, ::sub_x, 0].unsqueeze(1)
    u_test = u[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t]

    if xi is not None:
        xi_test = xi[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t].unsqueeze(1)
    else:
        xi_test = torch.zeros_like(u_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#===========================================================================
# Training functionalities
#===========================================================================

def train_nspde(model, train_loader, test_loader, device, myloss, batch_size=20, epochs=5000, learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20, time_train=False, time_eval=False):


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    losses_train = []
    losses_test = []

    times_train = [] 
    times_eval = []

    try:

        for ep in range(epochs):

            model.train()
            
            train_loss = 0.
            for u0_, xi_, u_ in train_loader:

                loss = 0.

                u0_ = u0_.to(device)
                xi_ = xi_.to(device)
                u_ = u_.to(device)

                t1 = default_timer()
                u_pred = model(u0_, xi_)
                loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                times_train.append(default_timer()-t1)

            test_loss = 0.
            with torch.no_grad():
                for u0_, xi_, u_ in test_loader:
                    
                    loss = 0.
                    
                    u0_ = u0_.to(device)
                    xi_ = xi_.to(device)
                    u_ = u_.to(device)

                    t1 = default_timer()

                    u_pred = model(u0_, xi_)

                    times_eval.append(default_timer()-t1)

                    loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                    test_loss += loss.item()

            scheduler.step()
            if ep % print_every == 0:
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Test Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        if time_train and time_eval:
            return model, losses_train, losses_test, times_train, times_eval 
        elif time_train and not time_eval:
            return model, losses_train, losses_test, times_train
        elif time_eval and not time_train:
            return model, losses_train, losses_test, times_eval 
        else:
            return model, losses_train, losses_test
        
    except KeyboardInterrupt:
        if time_train and time_eval:
            return model, losses_train, losses_test, times_train, times_eval 
        elif time_train and not time_eval:
            return model, losses_train, losses_test, times_train
        elif time_eval and not time_train:
            return model, losses_train, losses_test, times_eval 
        else:
            return model, losses_train, losses_test

#===============================================================================
# Plot solution at different time steps (1D)
#===============================================================================
def plot_1d(model, data_loader, device, i=1, T_=10, T=51, a=0):

    for u0_, xi_, u_ in data_loader:
        u0_ = u0_.to(device)
        xi_ = xi_.to(device)
        u_ = u_.to(device)
        break

    with torch.no_grad():
        u_pred = model(u0_,xi_)

    fig, ax = plt.subplots(1, T_, figsize=(T_*3, 3))

    times = np.linspace(a, T-1, T_)
    for j in range(T_):
        t = int(times[j])
        ax[j].plot(u_[i,...,t].detach().cpu().numpy(),label='true')
        ax[j].plot(u_pred[i,0,...,t].detach().cpu().numpy(),label='pred')
        ax[j].set_title(f'time step {j+1}')

    plt.legend()
    plt.tight_layout()
    plt.show()

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='bold', fontsize=34)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

def contour_plot_1d(model, data_loader, device, O_X, O_T, save_file=None):
    
    mpl.rcParams['xtick.major.pad'] = 8
    mpl.rcParams['ytick.major.pad'] = 8
    fig, ax = plt.subplots(2, 3, figsize=(20,10))
    x_m, t_m = np.meshgrid(O_T, O_X)

    for u0_, xi_, u_ in data_loader:
        u0_ = u0_.to(device)
        xi_ = xi_.to(device)
        u_ = u_.to(device)
        break

    with torch.no_grad():
        u_pred = model(u0_,xi_)

    for i in range(3):
        ax[0][i].contourf(x_m,t_m, u_[i].cpu().numpy(), 50, cmap=plt.cm.jet)
        ax[1][i].contourf(x_m,t_m, u_pred[i,0,...].cpu().numpy(), 50, cmap=plt.cm.jet)
        ax[0][i].set_xlabel('t')
        ax[0][i].set_ylabel('x')
        ax[1][i].set_xlabel('t')
        ax[1][i].set_ylabel('x')

    grid = plt.GridSpec(2, 3)
    create_subtitle(fig, grid[0, ::], 'Ground truth solutions')
    create_subtitle(fig, grid[1, ::], 'Predicted solutions with the Neural SPDE model')
    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file,bbox_inches='tight')

    plt.show()


#===============================================================================
# Utilities (adapted from https://github.com/zongyi-li/fourier_neural_operator)
#===============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# get grid
def get_grid(batch_size, dim_x, dim_y, dim_t=None):
    gridx = torch.linspace(0, 1, dim_x, dtype=torch.float)
    gridy = torch.linspace(0, 1, dim_y, dtype=torch.float)
    if dim_t:
        gridx = gridx.reshape(1, dim_x, 1, 1, 1).repeat([batch_size, 1, dim_y, dim_t, 1])
        gridy = gridy.reshape(1, 1, dim_y, 1, 1).repeat([batch_size, dim_x, 1, dim_t, 1])
        gridt = torch.linspace(0, 1, dim_t, dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, dim_t, 1).repeat([batch_size, dim_x, dim_y, 1, 1])
        return torch.cat((gridx, gridy, gridt), dim=-1).permute(0,4,1,2,3)
    gridx = gridx.reshape(1, dim_x, 1, 1).repeat([batch_size, 1, dim_y, 1])
    gridy = gridy.reshape(1, 1, dim_y, 1).repeat([batch_size, dim_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).permute(0,3,1,2)

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


###################
# The following utilities are for memory usage profiling
#
###################
def get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })

    return hook


def add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_backward_hook(generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)

def log_mem(model, inp, mem_log=None, exp=None, model_type='NSPDE'):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        add_memory_hooks(idx, module, mem_log, exp, hr)
        
    try:
        if model_type in ['NSPDE', 'NCDE']:
            out = model(inp[0], inp[1])
        else:
            out = model(inp)
        
        loss = out.sum()
        loss.backward()
  
    finally:
        [h.remove() for h in hr]

        return mem_log


def plot_mem(df, exps=None, normalize_call_idx=True, normalize_mem_all=True, filter_fwd=False, return_df=False, output_file=None):
    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots(figsize=(20, 10))
    for exp in exps:
        df_ = df[df.exp == exp]

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = df_.mem_all // 2 ** 20

        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[(df_["layer_idx"] == layer_idx) & (df_["hook_type"] == "fwd")]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"] <= callidx_stop]
            # df_ = df_[df_.call_idx < df_[df_.layer_idx=='bwd'].call_idx.min()]

        plot = df_.plot(ax=ax, x='call_idx', y='mem_all', label=exp)
        print('Maximum memory: {} MB'.format(df_['mem_all'].max()))
        if output_file:
            plot.get_figure().savefig(output_file)

    if return_df:
        return df_



###################
# The following utility returns the maximum memory usage
#
###################

def get_memory(device, reset=False, in_mb=True):
    if device is None:
        return float('nan')
    if device.type == 'cuda':
        if reset:
            torch.cuda.reset_max_memory_allocated(device)
        bytes = torch.cuda.max_memory_allocated(device)
        if in_mb:
            bytes = bytes / 1024 / 1024
        return bytes
    else:
        return float('nan')
