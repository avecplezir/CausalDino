"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.distributions.categorical import Categorical

from .gpt_utils import CfgNode as CN
import models.gpt_utils as tools
from .predictor import BatchNormGPT


# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, attn_type='causal', mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        assert attn_type in ['causal', 'all', 'id'], f'{attn_type} is not implemented!'

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attn_type == 'causal':
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        elif attn_type == 'id':
            mask = torch.eye(T).to(x.device)[None, None, :, :]
            att = att.masked_fill(mask == 0, float('-inf'))
        elif mask is not None:
            mask = mask[:, None, None, :]
            att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x, attn_type='causal', mask=None):
        x = x + self.attn(self.ln_1(x), attn_type=attn_type, mask=mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


def get_default_config():
    C = CN()
    # either model_type or (n_layer, n_head, n_embd) must be given in the config
    C.model_type = 'gpt'
    C.n_layer = None
    C.n_head = None
    C.n_embd = None
    # these options must be filled in externally
    C.vocab_size = None
    C.block_size = None
    # dropout hyperparameters
    C.embd_pdrop = 0.1
    C.resid_pdrop = 0.1
    C.attn_pdrop = 0.1
    return C


class GPT(nn.Module):
    """ GPT Language Model """

    def __init__(self, in_dim=256, out_dim=256, block_size=4,
                 model_type='gpt-micro-256', layer_norm=False,
                 maskemb=False, future_index=False, **kwargs):
        super().__init__()
        n_embd = out_dim
        config = get_default_config()
        config.block_size = block_size
        config.model_type = model_type
        assert config.block_size is not None
        self.block_size = config.block_size

        # translate from model_type to detailed configuration
        config.merge_from_dict({
                                   'gpt-mini': dict(n_layer=6, n_head=6, n_embd=n_embd),
                                   'gpt-micro': dict(n_layer=4, n_head=4, n_embd=n_embd),
                                   'gpt-nano': dict(n_layer=3, n_head=3, n_embd=n_embd),
                                   'gpt-micro-256-more': dict(n_layer=6, n_head=8, n_embd=n_embd),
                                   'gpt-micro-256': dict(n_layer=4, n_head=4, n_embd=n_embd),
                                   'gpt': dict(n_layer=8, n_head=8, n_embd=n_embd),
                                   'gpt-micro-256-half': dict(n_layer=2, n_head=4, n_embd=n_embd),
                               }[config.model_type])

        self.in_dim = in_dim
        self.input2nemb = nn.Linear(in_dim, config.n_embd) if self.in_dim != n_embd else None
        print('self.input2nemb', self.input2nemb)

        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        self.maskemb = maskemb
        if self.maskemb:
            print('use maskemb!')
            self.wme = nn.Embedding(1, config.n_embd)

        self.future_index = future_index
        if self.future_index:
            print('use future_index!')
            self.wfe = nn.Embedding(config.block_size, config.n_embd)

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.transformer.ln_f = nn.LayerNorm(config.n_embd)
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("gpt number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.Embedding):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, indices=None, attn_type='causal', mask=None, future_index=None, **kwargs):
        device = x.device
        b, t = x.size()[:2]

        if self.input2nemb:
            x = self.input2nemb(x) #map to correct emb dimention

        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) if indices is None else indices # shape (1, t)

        # forward the GPT model itself
        tok_emb = mask.unsqueeze(-1) * x if mask is not None else x # token embeddings of shape (b, t, n_embd), we zero embeddings where mask value is zero
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)

        if self.maskemb and attn_type != 'id':
            mask_emb = self.wme(torch.zeros_like(mask))
            tok_emb = tok_emb + (mask-1).unsqueeze(-1)*mask_emb #add mask emb where mask value is zero

        x = self.transformer.drop(tok_emb + pos_emb)

        if self.future_index:
            fi_emb = self.wfe(future_index.unsqueeze(1))
            x = torch.cat([fi_emb, x], 1)

        for block in self.transformer.h:
            x = block(x, attn_type=attn_type, mask=mask)

        if self.layer_norm:
            x = self.transformer.ln_f(x)

        return x


class GPTVAE(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, block_size=4,
                 model_type='gpt-micro-256', layer_norm=False,
                 maskemb=False, future_index=False, **kwargs):
        self.gpt = GPT(in_dim=in_dim, out_dim=out_dim, block_size=block_size,
                       model_type=model_type, layer_norm=layer_norm,
                       maskemb=maskemb, future_index=future_index, **kwargs)

        self._stoch = 32
        self._discrete = 32
        self._hidden = out_dim

        self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
        self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
        self.post2gpt = nn.Linear(self._hidden + self._stoch * self._discrete, self._hidden)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("gpt vae number of parameters: %.2fM" % (n_params / 1e6,))

    def forward(self, x, indices=None, **kwargs):
        # post
        x_post = self.gpt(x, indices=indices, attn_type='all')
        stats_post = self._suff_stats_layer('obs', x_post)

        # prior
        x_prior = self.gpt(x, indices=indices, attn_type='causal')
        stats_prior = self._suff_stats_layer('ims', x_prior)

        # sample posterior and transform it to feed to gpt
        stoch_post = self.get_dist(stats_post).sample()
        shape = list(stoch_post.shape[:-2]) + [self._stoch * self._discrete]
        stoch_post = stoch_post.reshape(shape)

        # prediction
        x_pred = self.post2gpt(torch.cat([x, stoch_post], -1))
        out = self.gpt(x_pred, indices=indices, attn_type='causal')

        return out, stoch_post, stats_post, stats_prior

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
