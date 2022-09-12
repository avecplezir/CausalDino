import torch
import torch.nn as nn
from torch import distributions as torchd

import torch.nn.functional as F
from utils.utils import trunc_normal_
import models.gpt_utils as tools


class DINOHead(nn.Module):
    def __init__(self, in_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return x


class MLPPredictor(nn.Module):
    def __init__(self, out_dim, emb_dim=768):
        super().__init__()
        self.wte = nn.Embedding(out_dim, emb_dim)
        self.mlp = DINOHead(emb_dim)
        self.register_buffer('indices', torch.arange(0, self.out_dim).unsqueeze(0))

    def forward(self, x):
        x = self.wte(x)
        x = self.mlp(x)
        return x

    def get_all(self, ):
        return self.forward(self.indices)


class OneLayerPredictor(nn.Module):
    def __init__(self, out_dim, emb_dim=768):
        super().__init__()
        self.out_dim = out_dim
        self.wte = nn.Embedding(out_dim, emb_dim)
        self.mlp = DINOHead(emb_dim, nlayers=1, bottleneck_dim=emb_dim, norm_last_layer=False)
        self.register_buffer('indices', torch.arange(0, self.out_dim).unsqueeze(0))

    def forward(self, x):
        x = self.wte(x)
        x = self.mlp(x)
        return x

    def get_all(self, ):
        return self.forward(self.indices)


class LinearPredictor(nn.Module):
    def __init__(self, out_dim, emb_dim=768):
        super().__init__()
        self.out_dim = out_dim
        self.wte = nn.Embedding(emb_dim)
        self.last_layer = nn.utils.weight_norm(nn.Linear(emb_dim, out_dim, bias=False))
        self.register_buffer('indices', torch.arange(0, self.out_dim).unsqueeze(0))

    def forward(self, x):
        x = self.wte(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    def get_all(self, ):
        return self.forward(self.indices)


class MLPfeaturePredictor(nn.Module):
    def __init__(self, n_embd=256, layer_norm=False, **kwargs):
        super().__init__()
        self.mlp = DINOHead(n_embd, bottleneck_dim=n_embd)
        self.layer_norm = layer_norm
        if self.layer_norm:
            print('layer norm in predictor!')
            self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, x, **kwargs):
        out = self.mlp(x)
        if self.layer_norm:
            out = self.ln_f(out)
        else:
            out = nn.functional.normalize(out, dim=-1, p=2)
        return out


class MLPPosPredictor(nn.Module):
    def __init__(self, n_embd=256, block_size=None, layer_norm=False, **kwargs):
        super().__init__()
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln_f = nn.LayerNorm(n_embd)

        self.predictor = DINOHead(2 * n_embd, bottleneck_dim=n_embd, nlayers=2)
        self.wpe = nn.Embedding(block_size, n_embd)

    def forward(self, x, indices=None, **kwargs):
        fp_emb = self.wpe(indices)
        out = self.predictor(torch.cat([fp_emb, x], -1))

        if self.layer_norm:
            out = self.ln_f(out)
        else:
            out = nn.functional.normalize(out, dim=-1, p=2)

        return out


class MLPPastPredictor(nn.Module):
    def __init__(self, n_embd=256, block_size=None, layer_norm=False, **kwargs):
        super().__init__()
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln_f = nn.LayerNorm(n_embd)

        self.predictor = DINOHead(2 * n_embd, bottleneck_dim=n_embd, nlayers=3)
        self.wpe = nn.Embedding(block_size, n_embd)

    def forward(self, x, indices=None, **kwargs):
        x = x[:, -1:].repeat(1, indices.size(1), 1)
        fp_emb = self.wpe(indices)
        out = self.predictor(torch.cat([fp_emb, x], -1))

        if self.layer_norm:
            out = self.ln_f(out)
        else:
            out = nn.functional.normalize(out, dim=-1, p=2)

        return out


def MLPBYOL(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


class Identity(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


class MLPVAEPredictor(nn.Module):
    def __init__(self, n_embd=256, block_size=None, layer_norm=False, **kwargs):
        super().__init__()
        self._stoch = 32
        self._discrete = 32
        self._hidden = n_embd
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln_f = nn.LayerNorm(n_embd)

        self.mlp_post = DINOHead(2 * n_embd, bottleneck_dim=self._hidden, nlayers=2)
        self.mlp_prior = DINOHead(2 * n_embd, bottleneck_dim=self._hidden, nlayers=2)

        self.predictor = DINOHead(n_embd + self._stoch * self._discrete, bottleneck_dim=self._hidden)
        self.wpe = nn.Embedding(block_size, n_embd)

        self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
        self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)

    def get_dist(self, state, dtype=None):
        logit = state['logit']
        dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
        return dist

    def _suff_stats_layer(self, name, x):
        if name == 'ims':
            x = self._ims_stat_layer(x)
        elif name == 'obs':
            x = self._obs_stat_layer(x)
        else:
            raise NotImplementedError
        logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
        return {'logit': logit}

    def forward(self, x, f_x=None, f_idx=None, **kwargs):

        t = x.size(1)
        f_x = f_x.unsqueeze(1).repeat(1, t, 1)
        f_idx = f_idx.unsqueeze(1).repeat(1, t)
        x_post = self.mlp_post(torch.cat([f_x, x], -1))
        stats_post = self._suff_stats_layer('obs', x_post)
        stoch_post = self.get_dist(stats_post).sample()

        fp_emb = self.wpe(f_idx)
        x_prior = self.mlp_prior(torch.cat([fp_emb, x], -1))
        stats_prior = self._suff_stats_layer('ims', x_prior)

        shape = list(stoch_post.shape[:-2]) + [self._stoch * self._discrete]
        stoch_post = stoch_post.reshape(shape)
        out = self.predictor(torch.cat([stoch_post, x], -1))

        if self.layer_norm:
            out = self.ln_f(out)
        else:
            out = nn.functional.normalize(out, dim=-1, p=2)

        return out, stoch_post, stats_post, stats_prior


class MLPVAE2FoldPredictor(nn.Module):
    def __init__(self, n_embd=256, block_size=4, layer_norm=False, **kwargs):
        super().__init__()
        self.wpe = nn.Embedding(block_size, n_embd)
        self.future_embgpt = MLPVAEPredictor(n_embd=n_embd, block_size=block_size, layer_norm=layer_norm, **kwargs)

    def forward(self, x, indices=None):
        pos = self.wpe(indices)
        x = x + pos
        return x


class HeadProba(nn.Module):
    def __init__(self, out_dim, emb_dim=256):
        super().__init__()
        self.last_layer = nn.utils.weight_norm(nn.Linear(emb_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        norm_last_layer = True
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.last_layer(x)
        return x


class MLPVAE2Predictor(nn.Module):
    def __init__(self, n_embd=256, block_size=None, layer_norm=False, **kwargs):
        super().__init__()
        self._stoch = 32
        self._discrete = 32
        self._hidden = n_embd
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln_f = nn.LayerNorm(n_embd)

        self.mlp_post = DINOHead(3 * n_embd, bottleneck_dim=self._hidden, nlayers=2)
        self.mlp_prior = DINOHead(2 * n_embd, bottleneck_dim=self._hidden, nlayers=2)

        self.predictor = DINOHead(n_embd + self._stoch * self._discrete, bottleneck_dim=self._hidden)
        self.wpe = nn.Embedding(block_size, n_embd)

        self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
        self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)

    def get_dist(self, state, dtype=None):
        logit = state['logit']
        dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
        return dist

    def _suff_stats_layer(self, name, x):
        if name == 'ims':
            x = self._ims_stat_layer(x)
        elif name == 'obs':
            x = self._obs_stat_layer(x)
        else:
            raise NotImplementedError
        logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
        return {'logit': logit}

    def forward(self, x, indices=None, f_x=None, **kwargs):

        t = x.size(1)
        f_x = f_x.repeat(1, t, 1)
        delta_pos = self.wpe(indices)
        x_post = self.mlp_post(torch.cat([f_x, x, delta_pos], -1))
        stats_post = self._suff_stats_layer('obs', x_post)
        stoch_post = self.get_dist(stats_post).sample()

        x_prior = self.mlp_prior(torch.cat([x, delta_pos], -1))
        stats_prior = self._suff_stats_layer('ims', x_prior)

        shape = list(stoch_post.shape[:-2]) + [self._stoch * self._discrete]
        stoch_post = stoch_post.reshape(shape)
        out = self.predictor(torch.cat([x, stoch_post], -1))

        if self.layer_norm:
            out = self.ln_f(out)
        else:
            out = nn.functional.normalize(out, dim=-1, p=2)

        return out, stoch_post, stats_post, stats_prior