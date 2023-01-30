__all__ = ['MultiCropWrapperGPT', 'MultiCropWrapperGPT1d']

from .base_wrapper import MultiCropWrapperBase


class MultiCropWrapperGPT(MultiCropWrapperBase):
    def forward_student(self, x_enc, indices, **kwargs):
        s_enc_logits = self.forward_encode(x_enc[:, 1:])
        s_pred_logits = self.forward_predict(x_enc[:, :-1], indices[:, :-1])
        return s_pred_logits, s_enc_logits

    def forward_teacher(self, x_enc, indices, **kwargs):
        return self.forward_student(x_enc, indices)


class MultiCropWrapperGPT1d(MultiCropWrapperBase):
    def forward_student(self, x_enc, indices, **kwargs):
        s_pred_logits = self.forward_predict(x_enc[:, :-1], indices[:, :-1])
        return s_pred_logits, None

    def forward_teacher(self, x_enc, indices, **kwargs):
        t_enc_logits = self.forward_encode(x_enc[:, 1:])
        return None, t_enc_logits
