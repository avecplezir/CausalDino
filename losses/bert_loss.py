__all__ = ['BertLoss', 'GPTLoss', 'TELoss', 'MemoryLoss']

import torch
import torch.nn.functional as F

from .feature_loss import FeatureLoss


class BertLoss(FeatureLoss):
    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_future_logits_list, masks = student_output
        t_enc_logits = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        CE_fe = self.compute_loss_fe(s_pred_future_logits_list, t_enc_proba, masks)
        total_loss = CE_fe

        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_future_logits_list, t_enc_proba, masks):
        total_loss = 0
        n_loss_terms = 0
        for s_pred_future_logits, mask in zip(s_pred_future_logits_list, masks):
            s_pred_future_log = F.log_softmax(s_pred_future_logits / self.student_temp, dim=-1)
            loss = -torch.sum(t_enc_proba * s_pred_future_log, dim=-1)
            inverse_mask = (~mask.bool()).long()
            n_terms = loss.size(0) * inverse_mask.sum()
            total_loss += (inverse_mask * loss).sum()
            n_loss_terms += n_terms

        total_loss /= n_loss_terms
        return total_loss


class GPTLoss(BertLoss):
    def compute_loss_fe(self, s_pred_future_logits, t_enc_proba, masks=None):
        s_pred_future_log = F.log_softmax(s_pred_future_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba[:, 1:] * s_pred_future_log[:, :-1], dim=-1)
        return loss.mean()


class TELoss(BertLoss):
    def compute_loss_fe(self, s_pred_future_logits_list, t_enc_proba, masks=None):
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


class MemoryLoss(FeatureLoss):
    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_future_logits, bert_mask = student_output
        t_enc_logits, memory_mask = teacher_output

        t = t_enc_logits.size(1)
        s_pred_future_logits = s_pred_future_logits[:, -t:]
        bert_mask = bert_mask[:, -t:]

        # print('memory loss')
        # print('s_pred_future_logits', s_pred_future_logits.shape)
        # print('bert_mask', bert_mask.shape)
        # print('memory_mask', memory_mask.shape)
        temp = self.teacher_temp_schedule[epoch]
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        inverse_bert_mask = (~bert_mask.bool()).long()
        inverse_mask = memory_mask * inverse_bert_mask
        # print('inverse_mask', inverse_mask.shape)
        # print('t_enc_proba', t_enc_proba.shape)
        # print('s_pred_future_logits', s_pred_future_logits.shape)

        CE_fe = self.compute_loss_fe(s_pred_future_logits, t_enc_proba, inverse_mask)
        total_loss = CE_fe

        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_future_logits, t_enc_proba, inverse_mask):
        s_pred_future_log = F.log_softmax(s_pred_future_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba * s_pred_future_log, dim=-1)
        n_terms = inverse_mask.sum() + 1e-16
        total_loss = (inverse_mask * loss).sum() / n_terms
        return total_loss