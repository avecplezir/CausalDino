import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import trunc_normal_


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
    def __init__(self, n_embd=256, **kwargs):
        super().__init__()
        self.mlp = DINOHead(n_embd)

    def forward(self, x, **kwargs):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return x


class MLPfeaturePredictorTimeEmb(nn.Module):
    def __init__(self, n_embd=256, block_size=None, **kwargs):
        super().__init__()
        self.mlp_post = DINOHead(2*n_embd)
        self.mlp_prior = DINOHead(2 * n_embd)
        z_dim = 16
        self.predictor = DINOHead(n_embd + z_dim)
        self.wpe = nn.Embedding(block_size, n_embd)

    def forward(self, x, f_x=None, f_idx=None, **kwargs):
        dist_post = self.mlp_post(torch.cat([f_x, x], 1))

        fp_emb = self.wpe(f_idx)
        dist_prior = self.mlp_prior(torch.cat([fp_emb, x], 1))

        z_post = dist_post.sample()

        out = self.predictor(torch.cat([z_post, x], 1))
        out = nn.functional.normalize(out, dim=-1, p=2)

        return out, dist_post, dist_prior


class MLPVAEPredictor(nn.Module):
    def __init__(self, n_embd=256, block_size=4, **kwargs):
        super().__init__()
        self.wpe = nn.Embedding(block_size, n_embd)
        self.future_embgpt = MLPfeaturePredictorTimeEmb(n_embd)

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


