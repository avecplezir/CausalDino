__all__ = ['VAELoss']

import torch
import torch.nn.functional as F
from torch import distributions as torchd

from .timeemb_loss import TimeEmbLoss
import models.gpt_utils as tools


class VAELoss(TimeEmbLoss):
    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc_logits, s_pred, _, s_indices = student_output
        t_enc_logits, t_pred, _, t_indices = teacher_output

        temp = self.teacher_temp_schedule[epoch]

        s_enc_proba = F.softmax(s_enc_logits / self.student_temp, dim=-1)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        CE_fe, kl = self.compute_loss_fe(s_pred, t_enc_proba, student, t_indices)
        CE_ef = self.compute_loss_ef(s_enc_proba, t_pred, teacher, t_indices, temp)

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef + self.args.kl_c * kl

        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'kl': kl,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred, t_enc_proba, student, indices):
        total_loss = 0
        n_loss_terms = 0
        total_kl_loss = 0
        # ip < ie
        for ie in range(1, self.n_global_views):  # future encoding
            s_pred_future, stoch_post, stats_post, stats_prior = \
                student.module.predictor.future_embgpt(s_pred[:, :ie], f_x=s_pred[:, ie], f_idx=indices[:, ie])
            s_pred_future_logits = student.module.headprob(student.module.head(s_pred_future))
            s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
            kl_loss = self.kl_loss(stats_post, stats_prior)
            total_kl_loss += kl_loss
            for ip in range(0, ie): #future_prediction from past
                loss = -torch.sum(t_enc_proba[:, ie] * torch.log(s_pred_future_proba[:, ip]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        total_kl_loss /= self.n_crops
        return total_loss, total_kl_loss

    def compute_loss_ef(self, s_enc_proba, t_pred, teacher, indices, temp):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ie in range(1, self.n_global_views):  # future encoding
            t_pred_future, _, _, _ = \
                teacher.predictor.future_embgpt(t_pred[:, :ie], f_x=t_pred[:, ie], f_idx=indices[:, ie])
            t_pred_future_logits = teacher.headprob(teacher.head(t_pred_future))
            t_pred_future_proba = F.softmax((t_pred_future_logits - self.center) / temp, dim=-1)
            for ip in range(0, ie): #future_prediction from past
                loss = -torch.sum(t_pred_future_proba[:, ip] * torch.log(s_enc_proba[:, ie]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    def get_dist(self, state, dtype=None):
        logit = state['logit']
        dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
        return dist

    def kl_loss(self, post, prior, forward=False, balance=0.8):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)

        value_lhs = kld(dist(lhs), dist(sg(rhs)))
        value_rhs = kld(dist(sg(lhs)), dist(rhs))

        loss = mix * value_lhs + (1 - mix) * value_rhs
        return loss.mean()