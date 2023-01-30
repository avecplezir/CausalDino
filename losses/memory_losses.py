__all__ = ['GPTMemoryLoss', 'DinoMemoryLoss', 'TEMemoryLoss', 'TEBertMemoryLoss']

import torch
import torch.nn.functional as F

from .feature_loss import FeatureLoss
from .base_losses import TEBertLoss


class GPTMemoryLoss(FeatureLoss):
    def forward(self, student_output: tuple, teacher_output: tuple, epoch: int, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_logits, s_enc_logits, m_mask, *_ = student_output
        t_pred_logits, t_enc_logits, *_ = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        CE_fe = self.compute_loss_fe(s_pred_logits, t_enc_logits, temp, m_mask) if self.args.CE_fe_c else 0.
        CE_ef = self.compute_loss_ef(s_enc_logits, t_pred_logits, temp, m_mask) if self.args.CE_ef_c else 0.

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        self.update_centers(t_enc_logits[:, -self.args.n_global_views:], t_pred_logits[:, -self.args.n_global_views:])
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)
        memory_size = m_mask.float().sum(-1).mean()

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'predict_entropy': self.entropy(self.predict_center),
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_logits, t_enc_logits, temp, mask):
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        s_pred_future_log = F.log_softmax(s_pred_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba * s_pred_future_log, dim=-1)
        loss = (mask * loss).sum()
        n_terms = mask.sum() + 1e-16
        loss = loss / n_terms
        return loss

    def compute_loss_ef(self, s_enc_logits, t_pred_logits, temp, mask):
        t_pred_proba = F.softmax((t_pred_logits - self.predict_center) / temp, dim=-1)
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_pred_proba * s_enc_log, dim=-1)
        loss = (mask * loss).sum()
        n_terms = mask.sum() + 1e-16
        loss = loss / n_terms
        return loss


class DinoMemoryLoss(FeatureLoss):
    def forward(self, student_output: tuple, teacher_output: tuple, epoch: int, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc_logits, *_ = student_output
        t_enc_logits, m_mask, *_ = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        CE_fe = self.dino_loss(s_enc_logits, t_enc_logits, temp, m_mask)

        total_loss = CE_fe

        self.update_centers(t_enc_logits[:, -self.args.n_global_views:], None)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)
        memory_size = m_mask.float().sum(-1).mean()

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'predict_entropy': self.entropy(self.predict_center),
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def dino_loss(self, s_enc_logits, t_enc_logits, temp, mask):
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        assert s_enc_log.size(1) == 1, 'support only one global view!'
        loss = -torch.sum(t_enc_proba * s_enc_log, dim=-1)
        loss = (mask * loss).sum()
        n_terms = mask.sum() + 1e-16
        loss /= n_terms
        return loss


class TEMemoryLoss(FeatureLoss):
    def forward(self, student_output: tuple, teacher_output: tuple, epoch: int, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_logits_list, s_pred_mask_list, s_enc_logits, *_ = student_output
        t_pred_logits_list, t_pred_mask_list, t_enc_logits, *_ = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        CE_fe = self.compute_loss_fe(s_pred_logits_list, t_enc_logits, temp, s_pred_mask_list) if self.args.CE_fe_c else 0.
        CE_ef = self.compute_loss_ef(s_enc_logits, t_pred_logits_list, temp, t_pred_mask_list) if self.args.CE_ef_c else 0.

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        self.update_centers(t_enc_logits, None)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)
        memory_size = t_pred_mask_list[-1].float().sum(-1).mean()

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'predict_entropy': self.entropy(self.predict_center),
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_logits_list, t_enc_logits, temp, s_mask_list):
        total_loss = 0
        n_terms = 1e-16
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        for s_pred_logits, mask in zip(s_pred_logits_list, s_mask_list):
            s_pred_log = F.log_softmax(s_pred_logits / self.student_temp, dim=-1)
            loss = -torch.sum(t_enc_proba * s_pred_log, dim=-1)
            total_loss += (mask * loss).sum()
            n_terms += mask.sum()

        total_loss = total_loss / n_terms
        return total_loss

    def compute_loss_ef(self, s_enc_logits, t_pred_logits_list, temp, t_mask_list):
        total_loss = 0
        n_terms = 1e-16
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)
        for t_pred_logits, mask in zip(t_pred_logits_list, t_mask_list):
            t_pred_proba = F.softmax((t_pred_logits - self.center) / temp, dim=-1)
            loss = -torch.sum(t_pred_proba * s_enc_log, dim=-1)
            total_loss += (mask * loss).sum()
            n_terms += mask.sum()

        total_loss = total_loss / n_terms
        return total_loss


class TEBertMemoryLoss(TEBertLoss):
    def forward(self, student_output, teacher_output, epoch: int, **kwargs):
        t_pred_logits_list, t_pred_mask, t_enc_logits, m_mask, m_enc = teacher_output
        memory_size = m_mask.float().sum(-1).mean()
        total_loss, dict = super().forward(student_output, teacher_output, epoch)
        dict['memory_size'] = memory_size
        return total_loss, dict
