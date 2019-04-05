import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import Parameter
from torch.autograd import Variable
import torchvision.datasets as dsets
import time
import click
import numpy
import numpy as np
import os
import random
from itertools import chain
import torch.nn.functional as F
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
import math


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self._shape = args
    def forward(self, x):
        return x.view(*self._shape)

def log_prob_gaussian(x, mu, log_vars, mean=False):
    lp = - 0.5 * math.log(2 * math.pi) \
        - log_vars / 2 - (x - mu) ** 2 / (2 * torch.exp(log_vars))
    if mean:
        return torch.mean(lp, -1)
    return torch.sum(lp, -1)

def log_prob_bernoulli(x, mu):
    lp = x * torch.log(mu + 1e-5) + (1. - y) * torch.log(1. - mu + 1e-5)
    return lp


def gaussian_kld(mu_left, logvar_left, mu_right, logvar_right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (torch.exp(logvar_left) / torch.exp(logvar_right)) +
                        ((mu_left - mu_right)**2.0 / torch.exp(logvar_right)) - 1.0)
    assert len(gauss_klds.size()) == 2
    return torch.sum(gauss_klds, 1)


class LayerNorm(nn.Module):
    def __init__(self, nb_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, nb_features))
        self.bias = nn.Parameter(torch.zeros(1, nb_features))

    def forward(self, x, gain=None, bias=None):
        assert len(x.size()) == 2
        if gain is None:
            gain = self.gain
        if bias is None:
            bias = self.bias
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + self.eps)
        z = (x - mean.expand_as(x)) / std.expand_as(x)
        return z * gain.expand_as(z) + bias.expand_as(z)


class LSTMCell(nn.Module):
    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_layernorm=False):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layernorm = use_layernorm
        self.has_bias = not self.use_layernorm
        if self.use_layernorm:
            self.use_bias = False
        print("LSTMCell: use_layernorm=%s" % use_layernorm)
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if self.use_layernorm:
            self.ln_ih = LayerNorm(4 * hidden_size)
            self.ln_hh = LayerNorm(4 * hidden_size)
        else:
            self.bias_ih = Parameter(torch.FloatTensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.FloatTensor(4 * hidden_size))
        self.init_weights()

    def init_weights(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        self.weight_ih.data.uniform_(-stdv, stdv)
        nn.init.orthogonal(self.weight_hh.data)
        if self.has_bias:
            self.bias_ih.data.fill_(0)
            self.bias_hh.data.fill_(0)

    def forward(self, input_, hx,
                gain_ih=None, gain_hh=None,
                bias_ih=None, bias_hh=None):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        assert input_.is_cuda
        h_0, c_0 = hx
        igates = torch.mm(input_, self.weight_ih)
        hgates = torch.mm(h_0, self.weight_hh)
        state = fusedBackend.LSTMFused()
        if self.use_layernorm:
            igates = self.ln_ih(igates, gain=gain_ih, bias=bias_ih)
            hgates = self.ln_hh(hgates, gain=gain_hh, bias=bias_hh)
            return state.apply(igates, hgates, c_0)
        else:
            return state.apply(igates, hgates, c_0,
                         self.bias_ih, self.bias_hh)

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LReLU(nn.Module):
    def __init__(self, c=1./3):
        super(LReLU, self).__init__()
        self.c = c

    def forward(self, x):
        return torch.clamp(F.leaky_relu(x, self.c), -3., 3.)


class ZForcing(nn.Module):
    def __init__(self, emb_dim, rnn_dim,
                 z_dim, mlp_dim, out_dim, out_type="gaussian",
                 cond_ln=False, nlayers=1, z_force=False, dropout=0.,
                 use_l2=False, drop_grad=False):
        super(ZForcing, self).__init__()
        assert not drop_grad, "drop_grad is not supported!"
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.rnn_dim = rnn_dim
        self.nlayers = nlayers
        self.z_dim = z_dim
        self.dropout = dropout
        self.out_type = out_type
        self.mlp_dim = mlp_dim
        self.cond_ln = cond_ln
        self.z_force = z_force
        self.use_l2 = use_l2
        self.drop_grad = drop_grad

        self.emb_mod = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            #nn.AvgPool2d(4),
            View(-1, 1024),
            nn.Linear(1024, emb_dim),
            nn.Dropout(dropout))

        self.bwd_mod = nn.LSTM(emb_dim, rnn_dim, nlayers)
        nn.init.orthogonal(self.bwd_mod.weight_hh_l0.data)
        self.fwd_mod = LSTMCell(
            emb_dim if cond_ln else emb_dim + mlp_dim,
            rnn_dim, use_layernorm=cond_ln)
        self.pri_mod = nn.Sequential(
            nn.Linear(rnn_dim, mlp_dim),
            LReLU(),
            nn.Linear(mlp_dim, z_dim * 2))
        self.inf_mod = nn.Sequential(
            nn.Linear(rnn_dim * 2, mlp_dim),
            LReLU(),
            nn.Linear(mlp_dim, z_dim * 2))
        if cond_ln:
            self.gen_mod = nn.Sequential(
                nn.Linear(z_dim, mlp_dim),
                LReLU(),
                nn.Linear(mlp_dim, 8 * rnn_dim))
        else:
            self.gen_mod = nn.Linear(z_dim, mlp_dim)
        self.aux_mod = nn.Sequential(
            nn.Linear(z_dim + rnn_dim, mlp_dim),
            LReLU(),
            nn.Linear(mlp_dim, 2 * rnn_dim))
        self.fwd_out_mod = nn.Linear(rnn_dim, out_dim)
        self.bwd_out_mod = nn.Linear(rnn_dim, out_dim)

    def save(self, filename):
        state = {
            'emb_dim': self.emb_dim,
            'rnn_dim': self.rnn_dim,
            'nlayers': self.nlayers,
            'mlp_dim': self.mlp_dim,
            'out_dim': self.out_dim,
            'out_type': self.out_type,
            'cond_ln': self.cond_ln,
            'z_force': self.z_force,
            'use_l2': self.use_l2,
            'z_dim': self.z_dim,
            'dropout': self.dropout,
            'drop_grad': self.drop_grad,
            'state_dict': self.state_dict()
        }
        torch.save(state, filename)

    @classmethod
    def load(cls, filename):
        state = torch.load(filename)
        model = ZForcing(
            state['inp_dim'], state['emb_dim'], state['rnn_dim'],
            state['z_dim'], state['mlp_dim'], state['out_dim'],
            nlayers=state['nlayers'], cond_ln=state['cond_ln'],
            out_type=state['out_type'], z_force=state['z_force'],
            use_l2=state.get('use_l2', False), drop_grad=state.get('drop_grad', False))
        model.load_state_dict(state['state_dict'])
        return model

    def reparametrize(self, mu, logvar, eps=None):
        std = logvar.mul(0.5).exp_()
        if eps is None:
            eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.rnn_dim).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.rnn_dim).zero_()))

    def fwd_pass(self, x_fwd, hidden, bwd_states=None, z_step=None):
        x_fwd_reshape = x_fwd.view(-1, *x_fwd.shape[2:])
        x_emb = self.emb_mod(x_fwd_reshape)
        x_fwd = x_emb.view(*x_fwd.shape[:2], self.emb_dim)
        nsteps = x_fwd.size(0)
        states = [(hidden[0][0], hidden[1][0])]
        klds, zs, log_pz, log_qz, aux_cs = [], [], [], [], []
        eps = Variable(next(self.parameters()).data.new(
            nsteps, x_fwd.size(1), self.z_dim).normal_())
        big = Variable(next(self.parameters()).data.new(x_fwd.size(1)).zero_()) + 0.5
        big = torch.bernoulli(big).unsqueeze(1)

        assert (z_step is None) or (nsteps == 1)
        for step in range(nsteps):
            states_step = states[step]
            x_step = x_fwd[step]
            h_step, c_step = states_step[0], states_step[1]
            r_step = eps[step]

            pri_params = self.pri_mod(h_step)
            pri_params = torch.clamp(pri_params, -8., 8.)
            pri_mu, pri_logvar = torch.chunk(pri_params, 2, 1)

            # inference phase
            if bwd_states is not None:
                b_step = bwd_states[step]
                inf_params = self.inf_mod(torch.cat((h_step, b_step), 1))
                inf_params = torch.clamp(inf_params, -8., 8.)
                inf_mu, inf_logvar = torch.chunk(inf_params, 2, 1)
                kld = gaussian_kld(inf_mu, inf_logvar, pri_mu, pri_logvar)
                z_step = self.reparametrize(inf_mu, inf_logvar, eps=r_step)
                if self.z_force:
                    h_step_ = h_step * 0.
                else:
                    h_step_ = h_step
                aux_params = self.aux_mod(torch.cat((h_step_, z_step), 1))
                aux_params = torch.clamp(aux_params, -8., 8.)
                aux_mu, aux_logvar = torch.chunk(aux_params, 2, 1)
                # disconnect gradient here
                b_step_ = b_step.detach()
                if self.use_l2:
                    aux_step = torch.sum((b_step_ - F.tanh(aux_mu)) ** 2.0, 1)
                else:
                    aux_step = -log_prob_gaussian(
                            b_step_, F.tanh(aux_mu), aux_logvar, mean=False)
            # generation phase
            else:
                # sample from the prior
                if z_step is None:
                    z_step = self.reparametrize(pri_mu, pri_logvar, eps=r_step)
                aux_step = torch.sum(pri_mu * 0., -1)
                inf_mu, inf_logvar = pri_mu, pri_logvar
                kld = aux_step

            i_step = self.gen_mod(z_step)
            if self.cond_ln:
                i_step = torch.clamp(i_step, -3, 3)
                gain_hh, bias_hh = torch.chunk(i_step, 2, 1)
                gain_hh = 1. + gain_hh
                h_new, c_new = self.fwd_mod(x_step, (h_step, c_step),
                                            gain_hh=gain_hh, bias_hh=bias_hh)
            else:
                h_new, c_new = self.fwd_mod(torch.cat((i_step, x_step), 1),
                                            (h_step, c_step))
            states.append((h_new, c_new))
            klds.append(kld)
            zs.append(z_step)
            aux_cs.append(aux_step)
            log_pz.append(log_prob_gaussian(z_step, pri_mu, pri_logvar))
            log_qz.append(log_prob_gaussian(z_step, inf_mu, inf_logvar))

        klds = torch.stack(klds, 0)
        aux_cs = torch.stack(aux_cs, 0)
        log_pz = torch.stack(log_pz, 0)
        log_qz = torch.stack(log_qz, 0)
        zs = torch.stack(zs, 0)

        outputs = [s[0] for s in states[1:]]
        outputs = torch.stack(outputs, 0)
        outputs = self.fwd_out_mod(outputs)
        return outputs, states[1:], klds, aux_cs, zs, log_pz, log_qz

    def infer(self, x, hidden):
        '''Infer latent variables for a given batch of sentences ``x''.
        '''
        x_ = x[:-1]
        y_ = x[1:]
        bwd_states, bwd_outputs = self.bwd_pass(x_, y_, hidden)
        fwd_outputs, fwd_states, klds, aux_nll, zs, log_pz, log_qz = self.fwd_pass(
                x_, hidden, bwd_states=bwd_states)
        return zs

    def bwd_pass(self, x, hidden):
        idx = np.arange(x.size(0))[::-1].tolist()
        idx = torch.LongTensor(idx)
        idx = Variable(idx).cuda()

        # invert the targets and revert back
        x_bwd = x.index_select(0, idx)
        # x_bwd = torch.cat([x_bwd, x[:1]], 0)
        x_bwd_reshape = x_bwd.view(-1, *x_bwd.shape[2:])
        print(x_bwd_reshape.shape)
        x_emb = self.emb_mod(x_bwd_reshape)
        x_bwd = x_emb.view(*x_bwd.shape[:2], self.emb_dim)
        states, _ = self.bwd_mod(x_bwd, hidden)
        outputs = self.bwd_out_mod(states)
        states = states.index_select(0, idx)
        outputs = outputs.index_select(0, idx)
        return states, outputs
    
    def generate_onestep(self, x_fwd, x_mask, hidden):
        nsteps, nbatch = x_fwd.size(0), x_fwd.size(1)
        #bwd_states, bwd_outputs = self.bwd_pass(x_bwd, hidden)
        fwd_outputs, fwd_states, klds, aux_nll, zs, log_pz, log_qz = self.fwd_pass(
            x_fwd, hidden, bwd_states=None)
        out_mu, out_logvar = torch.chunk(fwd_outputs, 2, -1) 
        hidden = (fwd_states[0][0].unsqueeze(0), fwd_states[0][1].unsqueeze(0))
        return (out_mu, out_logvar, hidden)
        
        '''kld = (klds * x_mask).sum(0)
        log_pz = (log_pz * x_mask).sum(0)
        log_qz = (log_qz * x_mask).sum(0)
        aux_nll = (aux_nll * x_mask).sum(0)
        if self.out_type == 'gaussian':
            out_mu, out_logvar = torch.chunk(fwd_outputs, 2, -1)
            fwd_nll = -log_prob_gaussian(y, out_mu, out_logvar)
            fwd_nll = (fwd_nll * x_mask).sum(0)
            out_mu, out_logvar = torch.chunk(bwd_outputs, 2, -1)
            bwd_nll = -log_prob_gaussian(y, out_mu, out_logvar)
            bwd_nll = (bwd_nll * x_mask).sum(0)
        elif self.out_type == 'softmax':
            fwd_out = fwd_outputs.view(nsteps * nbatch, self.out_dim)
            fwd_out = F.log_softmax(fwd_out)
            y = y.view(-1, 1)
            fwd_nll = torch.gather(fwd_out, 1, y).squeeze(1)
            fwd_nll = fwd_nll.view(nsteps, nbatch)
            fwd_nll = -(fwd_nll * x_mask).sum(0)
            bwd_out = bwd_outputs.view(nsteps * nbatch, self.out_dim)
            bwd_out = F.log_softmax(bwd_out)
            y = y.view(-1, 1)
            bwd_nll = torch.gather(bwd_out, 1, y).squeeze(1)
            bwd_nll = -bwd_nll.view(nsteps, nbatch)
            bwd_nll = (bwd_nll * x_mask).sum(0)

        if return_stats:
            return fwd_nll, bwd_nll, aux_nll, kld, log_pz, log_qz
        return fwd_nll.mean(), bwd_nll.mean(), aux_nll.mean(), kld.mean() '''
 

    def forward(self, x_fwd, x_bwd, y, x_mask, hidden, return_stats=False):
        nsteps, nbatch = x_fwd.size(0), x_fwd.size(1)
        bwd_states, bwd_outputs = self.bwd_pass(x_bwd, hidden)
        fwd_outputs, fwd_states, klds, aux_nll, zs, log_pz, log_qz = self.fwd_pass(
            x_fwd, hidden, bwd_states=bwd_states)
        kld = (klds * x_mask).sum(0)
        log_pz = (log_pz * x_mask).sum(0)
        log_qz = (log_qz * x_mask).sum(0)
        aux_nll = (aux_nll * x_mask).sum(0)
        if self.out_type == 'gaussian':
            out_mu, out_logvar = torch.chunk(fwd_outputs, 2, -1)
            fwd_nll = -log_prob_gaussian(y, out_mu, out_logvar)
            fwd_nll = (fwd_nll * x_mask).sum(0)
            out_mu, out_logvar = torch.chunk(bwd_outputs, 2, -1)
            bwd_nll = -log_prob_gaussian(y, out_mu, out_logvar)
            bwd_nll = (bwd_nll * x_mask).sum(0)
        elif self.out_type == 'softmax':
            fwd_out = fwd_outputs.view(nsteps * nbatch, self.out_dim)
            fwd_out = F.log_softmax(fwd_out)
            y = y.view(-1, 1)
            fwd_nll = torch.gather(fwd_out, 1, y).squeeze(1)
            fwd_nll = fwd_nll.view(nsteps, nbatch)
            fwd_nll = -(fwd_nll * x_mask).sum(0)
            bwd_out = bwd_outputs.view(nsteps * nbatch, self.out_dim)
            bwd_out = F.log_softmax(bwd_out)
            y = y.view(-1, 1)
            bwd_nll = torch.gather(bwd_out, 1, y).squeeze(1)
            bwd_nll = -bwd_nll.view(nsteps, nbatch)
            bwd_nll = (bwd_nll * x_mask).sum(0)

        if return_stats:
            return fwd_nll, bwd_nll, aux_nll, kld, log_pz, log_qz
        return fwd_nll.mean(), bwd_nll.mean(), aux_nll.mean(), kld.mean()
