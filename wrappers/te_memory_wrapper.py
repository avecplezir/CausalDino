__all__ = ['MultiCropWrapperTEMemory']

import torch
from .gpt_memory_wrapper import MultiCropWrapperGPTMemory


class MultiCropWrapperTEMemory(MultiCropWrapperGPTMemory):
    def forward_predict_te(self, x_enc, indices, mask):
        s_pred_logits_list = []
        s_pred_mask = []
        for ie in range(x_enc.size(1)-self.args.n_global_views, x_enc.size(1)):  # future encoding
            s_pred_logits = self.forward_predict(x_enc[:, :ie], future_index=indices[:, ie],
                                                 indices=indices[:, :ie], mask=mask[:, :ie])
            s_pred_logits_list.append(s_pred_logits[:, 1:])
            s_pred_mask.append(mask[:, :ie])
        return s_pred_logits_list, s_pred_mask

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        t_enc_logits = self.forward_encode(x_enc)

        self.memory.remove(video_indices)
        self.memory.add(x_enc)
        m_enc, m_mask = self.memory.retrieve()

        indices = self.get_indices(m_enc, maxlen=False, batch=True)
        t_pred_logits_list, t_pred_mask = self.forward_predict_te(m_enc, indices=indices, mask=m_mask)
        return t_pred_logits_list, t_pred_mask, t_enc_logits, m_mask, m_enc

    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        s_enc_logits = self.forward_encode(x_enc)

        x_m_enc = torch.cat([m_enc[:, :-x_enc.size(1)], x_enc], 1)
        indices = self.get_indices(m_enc, maxlen=False, batch=True)
        s_pred_logits_list, s_pred_mask = self.forward_predict_te(x_m_enc, indices=indices, mask=m_mask)
        return s_pred_logits_list, s_pred_mask, s_enc_logits
