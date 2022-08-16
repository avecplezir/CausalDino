__all__ = ['DINOTopkLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class DINOTopkLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, **kwargs):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.register_buffer("center", torch.zeros(1, 1, out_dim))
        self.register_buffer("predict_future_center", torch.zeros(1, 1, out_dim))
        self.register_buffer("predict_past_center", torch.zeros(1, 1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output.chunk(self.n_crops)
        s_enc_logits = torch.stack(student_out, 1)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = teacher_output.detach().chunk(self.n_crops)
        t_enc_logits = torch.stack(teacher_out, 1)

        indices = teacher_out.argmax(dim=-1)
        t_pred_future_logits = teacher.predictor(indices)
        s_pred_future_logits = student.module.predictor(indices)

        CE_fe = self.loss_fe(t_pred_future_logits, s_enc_logits, temp)
        CE_ef = self.loss_ef(t_enc_logits, s_pred_future_logits, temp)

        total_loss = 0.5*CE_fe + 0.5*CE_ef

        self.update_centers(t_enc_logits, t_pred_future_logits)
        entropy = -torch.sum(F.softmax(self.center, dim=-1) * F.log_softmax(self.center), dim=-1)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(student_out)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'entropy': entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def dirac_entropy(self, t_enc_logits):
        labels = torch.argmax(t_enc_logits, dim=-1)
        onehot = F.one_hot(labels)
        time_dirac_proba = onehot.float().mean(dim=1)
        dirac_entropy = -torch.sum(time_dirac_proba * torch.log(time_dirac_proba+1e-8), dim=-1).mean()
        max_entropy = np.log(onehot.size(1))
        dirac_entropy_proportion2max = dirac_entropy / max_entropy
        return dirac_entropy, dirac_entropy_proportion2max

    def loss_fe(self, t_pred_future_logits, s_enc_logits, teacher, temp):
        pred = (t_pred_future_logits[:, :-1] - self.center) / temp
        loss = torch.sum(-F.softmax(pred, dim=-1) * F.log_softmax(s_enc_logits[:, 1:] / self.student_temp, dim=-1), dim=-1)
        return loss.mean()

    def loss_ef(self, t_enc_logits, s_pred_future_logits, temp):
        encoding = (t_enc_logits[:, 1:] - self.predict_future_center) / temp
        loss = torch.sum(-F.softmax(encoding, dim=-1) * F.log_softmax(s_pred_future_logits[:, :-1] / self.student_temp, dim=-1), dim=-1)
        return loss.mean()

    @torch.no_grad()
    def update_centers(self, t_enc_logits, t_pred_future_logits, t_pred_past_logits):
        # update batch centers
        batch_center = self.get_batch_center(t_enc_logits)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        batch_center_pred_future = self.get_batch_center(t_pred_future_logits)
        self.predict_future_center = self.predict_future_center * self.center_momentum \
                                     + batch_center_pred_future * (1 - self.center_momentum)

        batch_center_pred_past = self.get_batch_center(t_pred_past_logits)
        self.predict_past_center = self.predict_past_center * self.center_momentum \
                                   + batch_center_pred_past * (1 - self.center_momentum)

    @torch.no_grad()
    def get_batch_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        b, t, *_ = teacher_output.shape
        batch_center = torch.sum(torch.sum(teacher_output, dim=0, keepdim=True), dim=1)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (b * t * dist.get_world_size())
        return batch_center
