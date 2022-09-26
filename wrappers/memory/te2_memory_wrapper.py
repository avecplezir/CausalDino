__all__ = ['MultiCropWrapperTEMemory']

import torch
from wrappers.base_wrapper import MultiCropWrapperBase


class MultiCropWrapperTEMemory(MultiCropWrapperBase):
    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        self.memory.add(x_enc)
        self.memory.remove(video_indices)
        m_enc, m_mask = self.memory.retrieve()
        t_enc_logits = self.forward_encode(x_enc)
        indices = self.get_indices(m_enc, maxlen=False)
        t_m_pred_logits = self.forward_predict(m_enc, indices=indices, mask=m_mask)
        t_pred_logits = t_m_pred_logits[:, -self.args.n_global_views:]
        return t_pred_logits, t_enc_logits, m_mask, m_enc

    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        s_enc_logits = self.forward_encode(x_enc)

        t = x_enc.size(1)
        x_m_enc = torch.cat([m_enc[:, :-t], x_enc], 1)
        indices = self.get_indices(x_m_enc, maxlen=False)
        s_pred_logits_list = []
        for ie in range(x_m_enc.size(1)-t, t):  # future encoding
            s_pred_logits = self.forward_predict(x_m_enc[:, :ie],
                                                 future_index=indices[:, ie],
                                                 indices=indices[:, :ie],
                                                 mask=m_mask)
            s_pred_logits_list.append(s_pred_logits[:, 1:])
        return s_pred_logits_list, s_enc_logits, m_mask