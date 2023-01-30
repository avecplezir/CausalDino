__all__ = ['GPTLoss', 'TELoss', 'TEBertLoss', 'TEBertLoss']

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .feature_loss import FeatureLoss


class GPTLoss(FeatureLoss):
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
        loss = -torch.sum(t_enc_proba * s_pred_future_log, dim=-1)
        return loss.mean()

    def compute_loss_ef(self, s_enc_logits, t_pred_logits, temp):
        t_pred_proba = F.softmax((t_pred_logits - self.predict_center) / temp, dim=-1)
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_pred_proba * s_enc_log, dim=-1)
        return loss.mean()


class TELoss(GPTLoss):
    def forward(self, student_output: tuple, teacher_output: tuple, epoch: int, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_logits_list, s_enc_logits = student_output
        t_pred_logits_list, t_enc_logits = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        CE_fe = self.compute_loss_fe(s_pred_logits_list, t_enc_logits, temp) if self.args.CE_fe_c else 0.
        CE_ef = self.compute_loss_ef(s_enc_logits, t_pred_logits_list, temp) if self.args.CE_ef_c else 0.

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        t_pred_cat_logits = torch.cat(t_pred_logits_list, 1)
        self.update_centers(t_enc_logits, t_pred_cat_logits)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'entropy': self.entropy(self.center),
                            'predict_entropy': self.entropy(self.predict_center),
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_logits_list, t_enc_logits, temp):
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        total_loss = 0
        n_loss_terms = 0
        for s_pred_logits in s_pred_logits_list:  # future encoding
            s_pred_log = F.log_softmax(s_pred_logits / self.student_temp, dim=-1)
            ie = s_pred_log.size(1)
            loss = -torch.sum(t_enc_proba[:, ie-1:ie] * s_pred_log, dim=-1)
            total_loss += loss.sum()
            n_loss_terms += loss.size(0) * ie

        total_loss /= n_loss_terms
        return total_loss

    def compute_loss_ef(self, s_enc_logits, t_pred_logits_list, temp):
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)

        total_loss = 0
        n_loss_terms = 0
        for t_pred_logits in t_pred_logits_list:  # future encoding
            t_pred_proba = F.softmax((t_pred_logits - self.predict_center) / temp, dim=-1)
            ie = t_pred_proba.size(1)
            loss = -torch.sum(t_pred_proba * s_enc_log[:, ie-1:ie], dim=-1)
            total_loss += loss.sum()
            n_loss_terms += loss.size(0) * ie

        total_loss /= n_loss_terms
        return total_loss


class TEBertLoss(FeatureLoss):
    def forward(self, student_output, teacher_output, epoch: int, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_pred_logits, s_enc_logits, inv_s_masks, *_ = student_output
        t_pred_logits, t_enc_logits, inv_t_masks, *_ = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        CE_fe = self.compute_loss_fe(s_pred_logits, t_enc_logits, temp, inv_s_masks) if self.args.CE_fe_c else 0.
        CE_ef = self.compute_loss_ef(s_enc_logits, t_pred_logits, temp, inv_t_masks) if self.args.CE_ef_c else 0.

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        self.update_center(t_enc_logits)
        self.update_predict_center(t_pred_logits, inv_t_masks) if self.args.CE_ef_c else None
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)
        mask_ratio = inv_s_masks.float().mean()

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'mask_ratio': mask_ratio,
                            'entropy': self.entropy(self.center),
                            'predict_entropy': self.entropy(self.predict_center),
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    @torch.no_grad()
    def update_center(self, t_enc_logits):
        batch_center = self.get_batch_center(t_enc_logits)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def update_predict_center(self, t_pred_logits, inv_t_masks):
        batch_center_pred = self.get_batch_predict_center_masked(t_pred_logits, inv_t_masks)
        self.predict_center = self.predict_center * self.center_momentum \
                                     + batch_center_pred * (1 - self.center_momentum)

    @torch.no_grad()
    def get_batch_predict_center_masked(self, t_pred_logits, inv_t_masks):
        b, t, *_ = t_pred_logits.shape
        batch_center = torch.sum(torch.sum(inv_t_masks.unsqueeze(-1)*t_pred_logits, dim=0, keepdim=True), dim=1,  keepdim=True)
        dist.all_reduce(batch_center)
        inv_m_sum = inv_t_masks.sum() + 1e-16
        dist.all_reduce(inv_m_sum)
        batch_center = batch_center / inv_m_sum
        return batch_center

    def compute_loss_fe(self, s_pred_logits, t_enc_logits, temp, inv_s_masks):
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        s_pred_log = F.log_softmax(s_pred_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba * s_pred_log, dim=-1)
        n_terms = inv_s_masks.sum() + 1e-16
        total_loss = (inv_s_masks * loss).sum()
        total_loss /= n_terms
        return total_loss

    def compute_loss_ef(self, s_enc_logits, t_pred_logits, temp, inv_t_masks):
        t_pred_proba = F.softmax((t_pred_logits - self.predict_center) / temp, dim=-1)
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_pred_proba * s_enc_log, dim=-1)
        n_terms = inv_t_masks.sum() + 1e-16
        total_loss = (inv_t_masks * loss).sum()
        total_loss /= n_terms
        return total_loss

