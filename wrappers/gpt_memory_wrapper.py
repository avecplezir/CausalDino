__all__ = ['MultiCropWrapperGPTMemory', 'MultiCropWrapperGPTMemory2',
           'MultiCropWrapperGPTMemory3', 'MultiCropWrapperGPTMemory4']

import torch
from wrappers.base_wrapper import MultiCropWrapperBase


class MultiCropWrapperGPTMemory(MultiCropWrapperBase):
    def get_indices(self, x, maxlen=True, batch=False):
        t = self.args.maxlen if maxlen else x.size(1)
        indices = torch.arange(t).flip([0]).unsqueeze(0).to(x.device)
        if batch:
            return indices.repeat((x.size(0)), 1)
        else:
            return indices

    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        t = x_enc.size(1)

        if self.args.scale_backbone_lr:
            scale_lr = self.args.scale_backbone_lr
        else:
            m_size = m_mask.sum().item()
            grad_size = t * x_enc.size(0)
            scale_lr = max(m_size, 1) / grad_size

        x_enc = x_enc * scale_lr
        x_enc.data.div_(scale_lr)
        x_m_enc = torch.cat([m_enc[:, :-t], x_enc], 1)

        s_m_enc_logits = self.forward_encode(x_m_enc[:, 1:])
        indices = self.get_indices(x_m_enc, maxlen=False)
        s_m_pred_logits = self.forward_predict(x_m_enc[:, :-1], indices=indices[:, :-1], mask=m_mask[:, :-1])
        return s_m_pred_logits, s_m_enc_logits, m_mask[:, :-1]

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        self.memory.remove(video_indices)
        self.memory.add(x_enc)
        m_enc, m_mask = self.memory.retrieve()

        t_m_enc_logits = self.forward_encode(m_enc[:, 1:])
        indices = self.get_indices(m_enc, maxlen=False)
        t_m_pred_logits = self.forward_predict(m_enc[:, :-1], indices=indices[:, :-1], mask=m_mask[:, :-1])
        return t_m_pred_logits, t_m_enc_logits, m_mask, m_enc


class MultiCropWrapperGPTMemory2(MultiCropWrapperGPTMemory):
    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        s_enc_logits = self.forward_encode(x_enc)

        t = x_enc.size(1)
        x_m_enc = torch.cat([m_enc[:, :-t], x_enc], 1)
        indices = self.get_indices(x_m_enc, maxlen=False)
        m_mask = m_mask[:, :-1]
        s_m_pred_logits = self.forward_predict(x_m_enc[:, :-1], indices=indices[:, :-1], mask=m_mask)
        s_pred_logits = s_m_pred_logits[:, -t:]
        m_mask = m_mask[:, -t:]
        return s_pred_logits, s_enc_logits, m_mask

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        t_enc_logits = self.forward_encode(x_enc)

        self.memory.remove(video_indices)
        self.memory.add(x_enc)
        m_enc, m_mask = self.memory.retrieve()

        indices = self.get_indices(m_enc, maxlen=False)
        t_m_pred_logits = self.forward_predict(m_enc[:, :-1], indices=indices[:, :-1], mask=m_mask[:, :-1])
        t = x_enc.size(1)
        t_pred_logits = t_m_pred_logits[:, -t:]
        return t_pred_logits, t_enc_logits, m_mask, m_enc


class MultiCropWrapperGPTMemory3(MultiCropWrapperGPTMemory):
    def forward_student(self, x_enc, video_indices=None, **kwargs):
        return self.forward_teacher(x_enc, video_indices=video_indices)

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        self.memory.remove(video_indices)
        self.memory.add(x_enc)
        m_enc, m_mask = self.memory.retrieve()

        t = x_enc.size(1)
        x_m_enc = torch.cat([m_enc[:, :-t], x_enc], 1)
        t_m_enc_logits = self.forward_encode(x_m_enc[:, 1:])
        indices = self.get_indices(m_enc, maxlen=False)
        t_m_pred_logits = self.forward_predict(x_m_enc[:, :-1], indices=indices[:, :-1], mask=m_mask[:, :-1])
        return t_m_pred_logits, t_m_enc_logits, m_mask[:, :-1], m_enc


class MultiCropWrapperGPTMemory4(MultiCropWrapperGPTMemory):
    def forward_student(self, x_enc, video_indices=None, **kwargs):
        return self.forward_teacher(x_enc, video_indices=video_indices)

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        t_enc_logits = self.forward_encode(x_enc)

        self.memory.remove(video_indices)
        self.memory.add(x_enc)
        m_enc, m_mask = self.memory.retrieve()

        t = x_enc.size(1)
        x_m_enc = torch.cat([m_enc[:, :-t], x_enc], 1)
        indices = self.get_indices(x_m_enc, maxlen=False)
        t_m_pred_logits = self.forward_predict(x_m_enc[:, :-1], indices=indices[:, :-1], mask=m_mask[:, :-1])
        t_pred_logits = t_m_pred_logits[:, -t:]
        m_mask = m_mask[:, -t:]
        return t_pred_logits, t_enc_logits, m_mask, m_enc