__all__ = ['GPT2Loss', 'GPT2MemoryLoss', 'TE2Loss']

import torch
import torch.nn.functional as F

from .feature_loss import FeatureLoss


class GPT2Loss(FeatureLoss):
    def forward(self, student_output: tuple, teacher_output: tuple, epoch: int, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_logits, s_enc_logits = student_output
        t_pred_logits, t_enc_logits = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        CE_fe = self.compute_loss_fe(s_pred_logits, t_enc_logits, temp) if self.args.CE_fe_c else 0.
        CE_ef = self.compute_loss_ef(s_enc_logits, t_pred_logits, temp) if self.args.CE_ef_c else 0.

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        self.update_centers(t_enc_logits, t_pred_logits)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'entropy': self.entropy(self.center),
                            'predict_entropy': self.entropy(self.predict_center),
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_logits, t_enc_logits, temp):
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        s_pred_future_log = F.log_softmax(s_pred_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba[:, 1:] * s_pred_future_log[:, :-1], dim=-1)
        return loss.mean()

    def compute_loss_ef(self, s_enc_logits, t_pred_logits, temp):
        t_pred_proba = F.softmax((t_pred_logits - self.predict_center) / temp, dim=-1)
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_pred_proba[:, :-1] * s_enc_log[:, 1:], dim=-1)
        return loss.mean()


class TE2Loss(GPT2Loss):
    def compute_loss_fe(self, s_pred_future_logits_list, t_enc_logits, temp):
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        total_loss = 0
        n_loss_terms = 0
        for s_pred_future_logits in s_pred_future_logits_list:  # future encoding
            s_pred_future_log = F.log_softmax(s_pred_future_logits / self.student_temp, dim=-1)
            ie = s_pred_future_log.size(1)
            loss = -torch.sum(t_enc_proba[:, ie:ie+1] * s_pred_future_log, dim=-1)
            total_loss += loss.sum()
            n_loss_terms += loss.size(0) * ie

        total_loss /= n_loss_terms
        return total_loss

    def compute_loss_ef(self, s_enc_logits, t_pred_future_logits_list, temp):
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)

        total_loss = 0
        n_loss_terms = 0
        for t_pred_future_logits in t_pred_future_logits_list:  # future encoding
            t_pred_proba = F.softmax((t_pred_future_logits - self.predict_center) / temp, dim=-1)
            ie = t_pred_proba.size(1)
            loss = -torch.sum(t_pred_proba * s_enc_log[:, ie:ie+1], dim=-1)
            total_loss += loss.sum()
            n_loss_terms += loss.size(0) * ie

        total_loss /= n_loss_terms
        return total_loss


class GPT2MemoryLoss(FeatureLoss):
    def forward(self, student_output: tuple, teacher_output: tuple, epoch: int, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_logits, s_enc_logits, m_mask = student_output
        t_pred_logits, t_enc_logits, *_ = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        CE_fe = self.compute_loss_fe(s_pred_logits, t_enc_logits, temp, m_mask) if self.args.CE_fe_c else 0.
        CE_ef = self.compute_loss_ef(s_enc_logits, t_pred_logits, temp, m_mask) if self.args.CE_ef_c else 0.

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        self.update_centers(t_enc_logits, t_pred_logits)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'entropy': self.entropy(self.center),
                            'predict_entropy': self.entropy(self.predict_center),
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_logits, t_enc_logits, temp, mask):
        mask = mask[:, :-1]
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        s_pred_future_log = F.log_softmax(s_pred_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba[:, 1:] * s_pred_future_log[:, :-1], dim=-1)
        loss = (mask * loss).sum()
        n_terms = mask.sum() + 1e-16
        loss = loss / n_terms
        return loss

    def compute_loss_ef(self, s_enc_proba, t_pred_future_proba, temp, mask):
        mask = mask[:, :-1]
        t_pred_proba = F.softmax((t_pred_future_proba - self.predict_center) / temp, dim=-1)
        s_enc_log = F.log_softmax(s_enc_proba / self.student_temp, dim=-1)
        loss = -torch.sum(t_pred_proba[:, :-1] * s_enc_log[:, 1:], dim=-1)
        loss = (mask * loss).sum()
        n_terms = mask.sum() + 1e-16
        loss = loss / n_terms
        return loss