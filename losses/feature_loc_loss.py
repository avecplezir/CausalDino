__all__ = ['FeatureLocLoss']

import torch
import torch.nn.functional as F

from .feature_loss import FeatureLoss


class FeatureLocLoss(FeatureLoss):
    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc_logits, s_pred_future_logits, s_pred_past_logits, s_indices = student_output
        t_enc_logits, t_pred_future_logits, t_pred_past_logits, t_indices = teacher_output

        temp = self.teacher_temp_schedule[epoch]

        s_enc_proba = F.softmax(s_enc_logits / self.student_temp, dim=-1)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)

        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        t_pred_future_proba = F.softmax((t_pred_future_logits - self.predict_future_center) / temp, dim=-1)

        s_enc_proba_g, s_enc_proba_l = s_enc_proba[:, :self.n_global_views], s_enc_proba[:, self.n_global_views:]
        t_enc_proba_g = t_enc_proba[:, :self.n_global_views]
        # s_pred_future_proba_g, s_pred_future_proba_l = s_pred_future_proba[:, :self.n_global_views], s_pred_future_proba[:, self.n_global_views:]
        # t_pred_future_proba_g, t_pred_future_proba_l = t_pred_future_proba[:, :self.n_global_views], t_pred_future_proba[:, self.n_global_views:]

        CE_fe = self.compute_loss_fe(s_pred_future_proba, t_enc_proba_g)
        CE_ef_g = self.compute_loss_ef(s_enc_proba_g, t_pred_future_proba)

        CE_ef_l = 0
        n = 0
        for i in range(int(len(s_enc_proba_l[0]) / self.n_global_views)):
            CE_ef_l += self.compute_loss_ef(s_enc_proba_l[:, self.n_global_views*i:self.n_global_views*(i+1)],
                                            t_pred_future_proba)
            n += 1

        CE_ef = (CE_ef_g + CE_ef_l) / (n+1)

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        self.update_centers(t_enc_logits, t_pred_future_logits, t_pred_past_logits)
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


