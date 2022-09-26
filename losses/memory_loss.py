__all__ = ['GPTMemoryLoss', 'MemoryLoss']

import torch
import torch.nn.functional as F

from .base.feature_loss import FeatureLoss
from losses.base.bert_loss import BertLoss


class GPTMemoryLoss(BertLoss):
    def forward(self, student_output: tuple, teacher_output: tuple, epoch: int, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_logits, memory_mask, *_ = student_output
        t_enc_logits, *_ = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        CE_fe = self.compute_loss_fe(s_pred_logits, t_enc_proba, memory_mask)

        total_loss = CE_fe

        memory_size = memory_mask.float().sum(-1).mean()
        self.update_centers(t_enc_logits[:, -self.n_global_views:], None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_logits, t_enc_proba, mask=None):
        mask = mask[:, :-1]
        s_pred_future_log = F.log_softmax(s_pred_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba[:, 1:] * s_pred_future_log[:, :-1], dim=-1)
        loss = (mask * loss).sum()
        n_terms = mask.sum() + 1e-16
        loss = loss / n_terms
        return loss


class MemoryLoss(FeatureLoss):
    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_m_pred_logits, bert_mask, s_enc_logits = student_output
        t_m_enc_logits, t_enc_logits, memory_mask, *_ = teacher_output

        t = t_m_enc_logits.size(1)
        s_m_pred_logits = s_m_pred_logits[:, -t:]
        bert_mask = bert_mask[:, -t:]

        temp = self.teacher_temp_schedule[epoch]
        t_m_enc_proba = F.softmax((t_m_enc_logits - self.center) / temp, dim=-1)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1) if self.args.CE_ee_c else 0.

        inverse_bert_mask = (~bert_mask.bool()).long()
        inverse_mask = memory_mask * inverse_bert_mask

        CE_fe = self.compute_loss_fe(s_m_pred_logits, t_m_enc_proba, inverse_mask)
        CE_ee = self.dino_loss(s_enc_logits, t_enc_proba) if self.args.CE_ee_c else 0.
        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ee_c * CE_ee

        self.update_centers(t_enc_logits, None)
        time_entropy = self.time_entropy(t_m_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_m_enc_logits)
        memory_size = memory_mask.float().sum(-1).mean()

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ee': CE_ee,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_m_pred_logits, t_m_enc_proba, inverse_mask):
        s_pred_log = F.log_softmax(s_m_pred_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_m_enc_proba * s_pred_log, dim=-1)
        n_terms = inverse_mask.sum() + 1e-16
        total_loss = (inverse_mask * loss).sum() / n_terms
        return total_loss

    def dino_loss(self, s_enc_logits, t_enc_proba):
        total_loss = 0
        n_loss_terms = 0
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)
        for iq in range(self.n_global_views):
            for v in range(self.n_crops):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = -torch.sum(t_enc_proba[:, iq] * s_enc_log[:, v], dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        n_loss_terms = max(1, n_loss_terms)
        total_loss /= n_loss_terms
        return total_loss
