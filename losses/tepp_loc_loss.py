__all__ = ['TEPPLocLoss']

import torch
import torch.nn.functional as F

from .timeemb_loss import TimeEmbLoss


class TEPPLocLoss(TimeEmbLoss):
    def compute_loss_fe(self, s_pred, t_enc_proba, student, indices):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ie in range(1, self.n_crops):  # future encoding
            s_pred_future = student.module.predictor.future_embgpt(s_pred[:, :ie], future_index=indices[:, ie])
            s_pred_future_logits = student.module.headprob(student.module.head(s_pred_future))
            s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
            for ip in range(0, ie): #future_prediction from past
                loss = -torch.sum(t_enc_proba[:, ie] * torch.log(s_pred_future_proba[:, ip]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
