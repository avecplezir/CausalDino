__all__ = ['TimeEmbLoss']

import torch
import torch.nn.functional as F

from .feature_loss import FeatureLoss


class TimeEmbLoss(FeatureLoss):
    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc_logits, s_pred, _, s_indices = student_output
        t_enc_logits, t_pred, _, t_indices = teacher_output

        temp = self.teacher_temp_schedule[epoch]

        s_enc_proba = F.softmax(s_enc_logits / self.student_temp, dim=-1)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        CE_fe = self.compute_loss_fe(s_pred, t_enc_proba, student, t_indices)
        CE_ef = self.compute_loss_ef(s_enc_proba, t_pred, teacher, t_indices, temp)

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred, t_enc_proba, student, indices):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ie in range(1, self.n_global_views):  # future encoding
            s_pred_future = student.module.predictor.future_embgpt(s_pred[:, :ie], future_index=indices[:, ie])
            s_pred_future_logits = student.module.headprob(s_pred_future)
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
            t_pred_future_logits = teacher.headprob(t_pred_future)
            t_pred_future_proba = F.softmax((t_pred_future_logits - self.center) / temp, dim=-1)
            for ip in range(0, ie): #future_prediction from past
                loss = -torch.sum(t_pred_future_proba[:, ip] * torch.log(s_enc_proba[:, ie]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
