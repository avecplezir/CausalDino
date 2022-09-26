__all__ = ['MultiCropWrapperGPT2Memory']

import torch
from .base_wrapper import MultiCropWrapperBase


class MultiCropWrapperGPT2Memory(MultiCropWrapperBase):
    def get_indices(self, x, maxlen=True):
        t = self.args.maxlen if maxlen else x.size(1)
        return torch.arange(t).flip([0]).unsqueeze(0).to(x.device)

    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        t = x_enc.size(1)

        if self.args.scale_backbone_lr:
            scale_lr = self.args.scale_backbone_lr
        else:
            m_size = m_mask.sum().item()
            grad_size = (t - 1) * x_enc.size(0)
            scale_lr = max(m_size, 1) / grad_size

        x_enc = x_enc * scale_lr
        x_enc.data.div_(scale_lr)
        x_m_enc = torch.cat([m_enc[:, :-t], x_enc], 1)

        s_m_enc_logits = self.forward_encode(x_m_enc[:, 1:])
        indices = self.get_indices(x_m_enc, maxlen=False)
        s_m_pred_logits = self.forward_predict(x_m_enc[:, :-1], indices=indices[:, :-1], mask=m_mask[:, :-1])
        return s_m_pred_logits, s_m_enc_logits, m_mask

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        self.memory.add(x_enc)
        self.memory.remove(video_indices)
        m_enc, m_mask = self.memory.retrieve()
        t_m_enc_logits = self.forward_encode(m_enc[:, 1:])
        indices = self.get_indices(m_enc, maxlen=False)
        t_m_pred_logits = self.forward_predict(m_enc[:, :-1], indices=indices, mask=m_mask)
        return t_m_pred_logits, t_m_enc_logits, m_mask, m_enc
