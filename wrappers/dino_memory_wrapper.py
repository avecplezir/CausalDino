__all__ = ['MultiCropWrapperDinoMemory']

from wrappers.base_wrapper import MultiCropWrapperBase


class MultiCropWrapperDinoMemory(MultiCropWrapperBase):
    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        s_enc_logits = self.forward_encode(x_enc)
        return s_enc_logits,

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        t_enc_head = self.forward_encode(x_enc, headprob=False)

        self.memory.remove(video_indices)
        self.memory.add(t_enc_head)
        m_enc_head, m_mask = self.memory.retrieve()

        t_m_enc_logits = self.headprob(m_enc_head)
        t = x_enc.size(1)
        return t_m_enc_logits[:, :-t], m_mask[:, :-t], m_enc_head
