__all__ = ['FeatureLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class FeatureLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, two_token=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.two_token = two_token
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc_logits, s_pred_future_logits, s_pred_past_logits = student_output
        t_enc_logits, t_pred_future_logits, t_pred_past_logits = teacher_output

        temp = self.teacher_temp_schedule[epoch]

        # Student # Encoding # Future Prediction # Past Prediction
        s_enc_proba = F.softmax(s_enc_logits / self.student_temp, dim=-1)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)[:, -1]

        # Teacher # Encoding # Future Prediction # Past Prediction
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)[:, 1:]
        t_pred_future_proba = F.softmax((t_pred_future_logits - self.center) / temp, dim=-1)

        marginal = F.softmax(self.center, dim=-1)
        CE_fe = self.compute_loss_fe(s_pred_future_proba, t_enc_proba)
        CE_ef = self.compute_loss_ef(t_pred_future_proba, s_enc_proba)
        KL_cm = self.compute_kl(s_enc_proba, marginal)

        total_loss = 0.9 * CE_fe + 0.1 * (CE_ef - KL_cm)

        batch_center = self.update_center(t_enc_logits)
        entropy = -torch.sum(marginal * torch.log(marginal), dim=-1)
        batch_entropy = -torch.sum(F.softmax(batch_center, dim=-1) * torch.log(marginal), dim=-1)
        time_events_proba = t_enc_proba.mean(1)
        time_entropy = -torch.sum(time_events_proba * torch.log(time_events_proba), dim=-1).mean()

        return total_loss, {'CE': total_loss, 'CE_ef': CE_ef, 'CE_fe': CE_fe, 'batch_entropy': batch_entropy, 'entropy': entropy,
                            'batch_time_entropy': time_entropy}

    def compute_kl(self, conditional, marginal):
        kl = conditional * (torch.log(conditional) - torch.log(marginal))
        return kl

    def compute_loss_fe(self, future_prediction, encoding):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ip in range(0, self.n_crops-2): #future_prediction from past
            for ie in range(ip + 1, self.n_crops-2): #future encoding
                loss = -torch.sum(encoding[:, ie] * torch.log(future_prediction[:, ip]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    def compute_loss_ef(self, encoding, future_prediction, past_prediction_sampled):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ip in range(0, self.n_crops-2): #future_prediction from past
            for ie in range(ip + 1, self.n_crops-2): #future encoding
                loss = -torch.sum(future_prediction[:, ip] * torch.log(encoding[:, ie]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return batch_center
