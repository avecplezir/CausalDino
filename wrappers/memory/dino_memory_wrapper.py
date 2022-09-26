__all__ = ['MultiCropWrapperDinoMemory']

import torch
from wrappers.base_wrapper import MultiCropWrapperBase


class MultiCropWrapperDinoMemory(MultiCropWrapperBase):
    def get_indices(self, x, maxlen=True):
        t = self.args.maxlen if maxlen else x.size(1)
        return torch.arange(t).flip([0]).unsqueeze(0).to(x.device)

    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        t = x_enc.size(1)
        x_m_enc = torch.cat([m_enc[:, :-t], x_enc], 1)
        s_m_enc_logits = self.forward_encode(x_m_enc)
        return s_m_enc_logits, m_mask

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        self.memory.add(x_enc)
        self.memory.remove(video_indices)
        m_enc, m_mask = self.memory.retrieve()
        t_m_enc_logits = self.forward_encode(m_enc[:, 1:])
        return t_m_enc_logits, m_mask, m_enc
