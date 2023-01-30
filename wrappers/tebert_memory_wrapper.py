__all__ = ['MultiCropWrapperTEBERTMemory',
           'MultiCropWrapperPastPredictionMemory',
           'MultiCropWrapperPastPredictionMemory2'
           ]

import torch

from .tebert_wrapper import MultiCropWrapperTEBERT


class MultiCropWrapperTEBERTMemory(MultiCropWrapperTEBERT):
    def get_indices(self, x, maxlen=True, batch=False):
        t = self.args.maxlen if maxlen else x.size(1)
        indices = torch.arange(t).flip([0]).unsqueeze(0).to(x.device)
        if batch:
            return indices.repeat((x.size(0)), 1)
        else:
            return indices

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        self.memory.remove(video_indices)
        self.memory.add(x_enc)
        m_enc, m_mask = self.memory.retrieve()

        t_enc_logits = self.forward_encode(m_enc)

        indices = self.get_indices(m_enc, maxlen=False, batch=True)
        m_bert_mask = self.generate_mask(m_enc)
        t_mask = m_bert_mask * m_mask
        t_pred_logits = self.forward_predict(m_enc, indices=indices, mask=t_mask, attn_type='all')
        inv_t_mask = (~t_mask.bool()).long() * m_mask
        return t_pred_logits, t_enc_logits, inv_t_mask, m_mask, m_enc

    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        x_m_enc = torch.cat([m_enc[:, :-x_enc.size(1)], x_enc], 1)
        s_enc_logits = self.forward_encode(x_m_enc)

        indices = self.get_indices(x_m_enc, maxlen=False, batch=True)
        x_m_bert_mask = self.generate_mask(x_m_enc)
        s_mask = x_m_bert_mask * m_mask
        s_pred_logits = self.forward_predict(x_m_enc, indices=indices, mask=s_mask, attn_type='all')
        inv_s_mask = (~s_mask.bool()).long() * m_mask
        return s_pred_logits, s_enc_logits, inv_s_mask


class MultiCropWrapperPastPredictionMemory(MultiCropWrapperTEBERTMemory):
    def generate_mask(self, x):
        b, T, *_ = x.size()
        masks = torch.zeros(1, T)
        masks[:, -1] = 1
        masks = masks.long().repeat(b, 1).to(x.device)
        return masks


class MultiCropWrapperPastPredictionMemory2(MultiCropWrapperPastPredictionMemory):
    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        x_enc_head = self.forward_encode(x_enc, headprob=False)

        self.memory.remove(video_indices)
        self.memory.add(x_enc_head)
        m_enc_head, m_mask = self.memory.retrieve()

        t_enc_logits = self.headprob(m_enc_head)

        m_enc = torch.cat([torch.zeros_like(x_enc[:, :1].repeat(1, m_enc_head.size(1)-1, 1)), x_enc], 1)
        indices = self.get_indices(m_enc, maxlen=False, batch=True)
        m_bert_mask = self.generate_mask(m_enc)
        t_mask = m_bert_mask * m_mask
        t_pred_logits = self.forward_predict(m_enc, indices=indices, mask=t_mask, attn_type='all')
        inv_t_mask = (~t_mask.bool()).long() * m_mask
        return t_pred_logits, t_enc_logits, inv_t_mask, m_mask, m_enc_head

    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        x_enc_head = self.forward_encode(x_enc, headprob=False)
        x_m_enc_head = torch.cat([m_enc[:, :-x_enc.size(1)], x_enc_head], 1)
        s_enc_logits = self.headprob(x_m_enc_head)

        x_m_enc = torch.cat([torch.zeros_like(x_enc[:, :1].repeat(1, m_enc.size(1) - 1, 1)), x_enc], 1)
        indices = self.get_indices(x_m_enc, maxlen=False, batch=True)
        x_m_bert_mask = self.generate_mask(x_m_enc)
        s_mask = x_m_bert_mask * m_mask
        s_pred_logits = self.forward_predict(x_m_enc, indices=indices, mask=s_mask, attn_type='all')
        inv_s_mask = (~s_mask.bool()).long() * m_mask
        return s_pred_logits, s_enc_logits, inv_s_mask

