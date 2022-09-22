__all__ = ['VAELoss', 'MemoryVAELoss']

import torch
import torch.nn.functional as F
from torch import distributions as torchd

from .bert_loss import BertLoss
import models.gpt_utils as tools


class VAELoss(BertLoss):
    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_logits, stats_post, stats_prior = student_output
        t_enc_logits, *_ = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        CE_fe, kl = self.compute_loss_fe(s_pred_logits, t_enc_proba, stats_post, stats_prior)
        total_loss = self.args.CE_fe_c * CE_fe + self.args.kl_c * kl

        self.update_centers(t_enc_logits, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'kl': kl,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_logits, t_enc_proba, stats_post, stats_prior):
        s_pred_log = F.log_softmax(s_pred_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba[:, -1:] * s_pred_log, dim=-1)
        total_loss = loss.mean()
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


class MemoryVAELoss(VAELoss):
    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_m_pred_logits, bert_mask, s_enc_logits, stats_post, stats_prior = student_output
        t_m_enc_logits, t_enc_logits, memory_mask, *_ = teacher_output

        t = t_m_enc_logits.size(1)
        s_m_pred_logits = s_m_pred_logits[:, -t:]
        bert_mask = bert_mask[:, -t:]

        temp = self.teacher_temp_schedule[epoch]
        t_m_enc_proba = F.softmax((t_m_enc_logits - self.center) / temp, dim=-1)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1) if self.args.CE_ee_c else 0.

        inverse_bert_mask = (~bert_mask.bool()).long()
        inverse_mask = memory_mask * inverse_bert_mask

        CE_fe, kl = self.compute_loss_fe(s_m_pred_logits, t_m_enc_proba, stats_post, stats_prior, inverse_mask)
        CE_ee = self.dino_loss(s_enc_logits, t_enc_proba) if self.args.CE_ee_c else 0.
        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ee_c * CE_ee + self.args.kl_c * kl

        self.update_centers(t_enc_logits, None)
        time_entropy = self.time_entropy(t_m_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_m_enc_logits)
        memory_size = memory_mask.float().sum(-1).mean()

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ee': CE_ee,
                            'kl': kl,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_logits, t_enc_proba, stats_post, stats_prior, memory_mask):
        s_pred_log = F.log_softmax(s_pred_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba[:, -1:] * s_pred_log, dim=-1)
        kl_loss = self.kl_loss(stats_post, stats_prior, balance=self.args.kl_balance)
        mask_sum = memory_mask.sum() + 1e-16
        total_loss = (memory_mask * loss).sum() / mask_sum
        kl_loss = (kl_loss * loss).sum() / mask_sum
        return total_loss, kl_loss

