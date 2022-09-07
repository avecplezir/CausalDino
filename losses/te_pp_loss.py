__all__ = ['TEPPLoss']

import torch
import torch.nn.functional as F

from .timeemb_loss import TimeEmbLoss


class TEPPLoss(TimeEmbLoss):
    def compute_loss_fe(self, s_pred, t_enc_proba, student, indices):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ie in range(1, self.n_global_views):  # future encoding
            s_pred_future = student.module.predictor.future_embgpt(s_pred[:, :ie], future_index=indices[:, ie])
            s_pred_future_logits = student.module.headprob(student.module.head(s_pred_future))
            s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
            for ip in range(0, ie): #future_prediction from past
                loss = -torch.sum(t_enc_proba[:, ie] * torch.log(s_pred_future_proba[:, ip]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    def compute_loss_ef(self, s_enc_proba, t_pred, teacher, indices, temp):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ie in range(1, self.n_global_views):  # future encoding
            t_pred_future = teacher.predictor.future_embgpt(t_pred[:, :ie], future_index=indices[:, ie])
            t_pred_future_logits = teacher.headprob(teacher.head(t_pred_future))
            t_pred_future_proba = F.softmax((t_pred_future_logits - self.center) / temp, dim=-1)
            for ip in range(0, ie): #future_prediction from past
                loss = -torch.sum(t_pred_future_proba[:, ip] * torch.log(s_enc_proba[:, ie]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
