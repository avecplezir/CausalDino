__all__ = ['MultiCropWrapperGPT2']

from .base_wrapper import MultiCropWrapperBase


class MultiCropWrapperGPT2(MultiCropWrapperBase):
    def forward_student(self, x_enc, indices, **kwargs):
        s_enc_logits = self.forward_encode(x_enc[:, 1:])
        s_pred_logits = self.forward_predict(x_enc[:, :-1], indices[:, :-1])
        print('s_enc_logits', s_enc_logits.shape)
        print('s_pred_logits', s_pred_logits.shape)
        return s_pred_logits, s_enc_logits

    def forward_teacher(self, x_enc, indices, **kwargs):
        self.forward_student(x_enc, indices)

