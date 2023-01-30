__all__ = ['MultiCropWrapperTEBERT']

import numpy as np
import torch

from .base_wrapper import MultiCropWrapperBase


class MultiCropWrapperTEBERT(MultiCropWrapperBase):
    def generate_mask(self, x):
        b, T, *_ = x.size()
        T = T-1
        masks = torch.tril(torch.ones(T, T), diagonal=-1)
        masks = torch.cat([masks, torch.ones(T, 1)], 1).long()
        r_indices = np.random.choice(range(len(masks)), size=b, replace=True)
        masks = masks[r_indices]
        return torch.tensor(masks).to(x.device)

    def forward_student(self, x_enc, indices=None, mask=None, **kwargs):
        s_mask = self.generate_mask(x_enc)
        s_enc_logits = self.forward_encode(x_enc)
        s_pred_logits = self.forward_predict(x_enc, indices=indices, mask=s_mask, attn_type='all')
        inv_s_mask = (~s_mask.bool()).long()
        return s_pred_logits, s_enc_logits, inv_s_mask

    def forward_teacher(self, x_enc, indices=None, mask=None, **kwargs):
        return self.forward_student(x_enc, indices=indices, mask=mask)
