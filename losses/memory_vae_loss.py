__all__ = ['MemoryVAELoss']

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import distributions as torchd

import models.gpt_utils as tools


class MemoryVAELoss(nn.Module):
    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        s_enc_logits, _, _ = student_output
        t_enc_logits, _, _ = teacher_output


        temp = self.teacher_temp_schedule[epoch]

        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        CE_fe, kl = self.compute_loss_fe(s_memory_enc, memory_mask, t_enc_proba,)

        total_loss = self.args.CE_fe_c * CE_fe + self.args.kl_c * kl
        memory_size = memory_mask.sum(-1).mean()

        self.add_memory(t_enc[:, 0])
        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'KL': kl,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }


    def compute_loss_fe(self, memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices):
        memory_mask[:, -1] = 0
        s_pred_future, stoch_post, stats_post, stats_prior = \
        student.module.predictor(memory_enc, indices=pos_indices, f_x=memory_enc[:, -1:])
        s_pred_future_logits = student.module.head(s_pred_future)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
        print('t_enc_proba', t_enc_proba.shape, 's_pred_future_proba', s_pred_future_proba.shape)
        loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
        mask_sum = memory_mask.sum() + 1e-16
        total_loss = (memory_mask * loss).sum() / mask_sum

        kl_loss = self.kl_loss(stats_post, stats_prior, balance=self.args.kl_balance)

        return total_loss, kl_loss

    def get_dist(self, state):
        logit = state['logit']
        dist = torchd.independent.Independent(tools.OneHotDist(logit), 1)
        return dist

    def kl_loss(self, post, prior, balance=0.8):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = (post, prior)
        mix = 1 - balance

        value_lhs = kld(dist(lhs), dist(sg(rhs)))
        value_rhs = kld(dist(sg(lhs)), dist(rhs))

        loss = mix * value_lhs + (1 - mix) * value_rhs
        return loss.mean()