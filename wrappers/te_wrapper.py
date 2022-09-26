__all__ = ['MultiCropWrapperTE']

from .base_wrapper import MultiCropWrapperBase


class MultiCropWrapperTE(MultiCropWrapperBase):
    def forward_student(self, x_enc, indices=None, **kwargs):
        s_enc_logits = self.forward_encode(x_enc[:, 1:])

        s_pred_logits_list = []
        for ie in range(1, self.args.n_global_views):  # future encoding
            s_pred_logits = self.forward_predict(x_enc[:, :ie], future_index=indices[:, ie], indices=indices[:, :ie])
            s_pred_logits_list.append(s_pred_logits[:, 1:])
        return s_pred_logits_list, s_enc_logits

    def forward_teacher(self, x_enc, indices=None, **kwargs):
        return self.forward_student(x_enc, indices=indices)
