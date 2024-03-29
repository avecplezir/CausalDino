import torch
import torch.nn as nn
from torch import distributions as torchd

import torch.nn.functional as F
from utils.utils import trunc_normal_
import models.gpt_utils as tools


class DINOHead(nn.Module):
    def __init__(self, in_dim=0, out_dim=0, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256, skip_last=False, layer_norm=False, l2norm=False, use_ln=False, **kwargs):
        super().__init__()
        self.skip_last = skip_last
        self.layer_norm = layer_norm
        self.l2norm = l2norm
        print('layer_norm in dinohead', layer_norm)
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                print('dinohead use_bn!')
                layers.append(nn.BatchNorm1d(hidden_dim))
            if use_ln:
                print('dinohead use_ln!')
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                if use_ln:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        if self.layer_norm:
            self.ln_f = nn.LayerNorm(bottleneck_dim)

        self.apply(self._init_weights)
        if not skip_last:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, **kwargs):
        reshape = (len(x.size()) == 3)
        if reshape:
            b, t, emb = x.size()
            x = x.reshape(b * t, emb)
        x = self.mlp(x)
        if reshape:
            x = x.reshape(b, t, -1)

        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)

        return x


class BatchNormGPT(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        reshape = (len(x.size()) == 3)
        if reshape:
            b, t, emb = x.size()
            x = x.reshape(b * t, emb)
        x = self.bn(x)
        if reshape:
            x = x.reshape(b, t, -1)
        return x


class Projector(DINOHead):
    def __init__(self, in_dim=0, out_dim=256, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                layer_norm=False, l2norm=False, use_ln=False, **kwargs):
        super().__init__(in_dim=in_dim, out_dim=0, use_bn=use_bn, norm_last_layer=norm_last_layer,
                         nlayers=nlayers, hidden_dim=hidden_dim, bottleneck_dim=256,
                         skip_last=True, layer_norm=layer_norm, l2norm=l2norm, use_ln=use_ln)

    def forward(self, x, **kwargs):
        reshape = (len(x.size()) == 3)
        if reshape:
            b, t, emb = x.size()
            x = x.reshape(b * t, emb)
        x = self.mlp(x)
        if reshape:
            x = x.reshape(b, t, -1)

        if self.layer_norm:
            x = self.ln_f(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=-1, p=2)

        return x


class MLPBYOL(nn.Module):
    def __init__(self, in_dim, out_dim=256, hidden_dim=4096, layer_norm=None,
                 l2norm=None, use_bn=True, **kwargs):
        super().__init__()
        out_dim = 256
        self.layer_norm = layer_norm
        self.l2norm = l2norm
        if self.layer_norm:
            self.ln_f = nn.LayerNorm(in_dim)

        if use_bn:
            print('predictor use_bn!')
            self.mlp = nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, out_dim)
                    )
        else:
            self.mlp = nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, out_dim)
                    )

    def forward(self, x, **kwargs):
        out = x
        if len(x.size()) == 3:
            b, t, emb = x.size()
            out = out.reshape(b * t, emb)
        out = self.mlp(out)
        if len(x.size()) == 3:
            out = out.reshape(b, t, -1)

        if self.layer_norm:
            out = self.ln_f(out)

        if self.l2norm:
            out = nn.functional.normalize(out, dim=-1, p=2)

        return out


class MLPPosPredictor(nn.Module):
    def __init__(self, n_embd=256, block_size=None, layer_norm=False, use_bn=False, **kwargs):
        super().__init__()
        self.predictor = Projector(2 * n_embd, bottleneck_dim=n_embd, nlayers=2, use_bn=use_bn, layer_norm=layer_norm)
        self.wpe = nn.Embedding(block_size, n_embd)

    def forward(self, x, indices=None, **kwargs):
        fp_emb = self.wpe(indices)
        out = self.predictor(torch.cat([fp_emb, x], -1))
        return out


class MLPPastPredictor(nn.Module):
    def __init__(self, n_embd=256, block_size=None, layer_norm=False, use_bn=False, **kwargs):
        super().__init__()
        self.predictor = Projector(2 * n_embd, bottleneck_dim=n_embd, nlayers=3, use_bn=use_bn, layer_norm=layer_norm)
        self.wpe = nn.Embedding(block_size, n_embd)

    def forward(self, x, indices=None, **kwargs):
        x = x[:, -1:].repeat(1, indices.size(1), 1)
        fp_emb = self.wpe(indices)
        out = self.predictor(torch.cat([fp_emb, x], -1))

        return out


class Identity(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


class MLPVAEPredictor(nn.Module):
    def __init__(self, n_embd=256, block_size=None, layer_norm=False,
                 use_bn=False, hidden_dim=2048, **kwargs):
        super().__init__()
        self._stoch = 32
        self._discrete = 32
        self._hidden = n_embd
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln_f = nn.LayerNorm(n_embd)

        self.mlp_post = DINOHead(2 * n_embd, bottleneck_dim=self._hidden, nlayers=2, use_bn=use_bn, hidden_dim=hidden_dim)
        self.mlp_prior = DINOHead(2 * n_embd, bottleneck_dim=self._hidden, nlayers=2, use_bn=use_bn, hidden_dim=hidden_dim)

        self.predictor = DINOHead(n_embd + self._stoch * self._discrete, bottleneck_dim=self._hidden,
                                  use_bn=use_bn, hidden_dim=hidden_dim)
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


class HeadProbal2Norm(nn.Module):
    def __init__(self, out_dim, emb_dim=256):
        super().__init__()
        self.last_layer = nn.utils.weight_norm(nn.Linear(emb_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        norm_last_layer = True
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class HeadProbal2NormDp(nn.Module):
    def __init__(self, out_dim, emb_dim=256):
        super().__init__()
        self.last_layer = nn.utils.weight_norm(nn.Linear(emb_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        norm_last_layer = True
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class L2Norm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        x = nn.functional.normalize(x, dim=-1, p=2)
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