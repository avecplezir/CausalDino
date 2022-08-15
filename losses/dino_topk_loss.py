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
        self.register_buffer("center", torch.zeros(1, out_dim))
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
        student_out = torch.stack(student_out, 1)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = teacher_output.detach().chunk(self.n_crops)
        teacher_out = torch.stack(teacher_out, 1)

        CE_fe = self.loss_fe(teacher_out, student_out, teacher, temp)
        CE_ef = self.loss_ef(teacher_out, student_out, student, temp)

        batch_center = self.update_center(teacher_output)
        entropy = -torch.sum(F.softmax(self.center, dim=-1) * F.log_softmax(self.center), dim=-1)
        batch_entropy = -torch.sum(F.softmax(batch_center, dim=-1) * F.log_softmax(self.center), dim=-1)
        s_enc_logits = torch.stack(student_out, 1)
        time_events_proba = F.softmax(s_enc_logits, dim=-1).mean(1)
        time_entropy = -torch.sum(time_events_proba * torch.log(time_events_proba), dim=-1).mean()

        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(s_enc_logits)

        return total_loss, {'CE': total_loss,
                            'batch_entropy': batch_entropy,
                            'entropy': entropy,
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            'temp': temp,
                            }

    def loss_fe(self, teacher_out, student_out, teacher, temp):
        indices = teacher_out.argmax(dim=-1)
        print('indices', indices.shape)
        pred = teacher.predictor(indices)
        print('pred', pred.shape)
        pred = (pred[:, :-1] - teacher_out[:, :-1]) / temp
        loss = torch.sum(-F.softmax(pred, dim=-1) * F.log_softmax(student_out[:, 1:] / self.student_temp, dim=-1), dim=-1)
        return loss.mean()

    def loss_ef(self, teacher_out, student_out, student, temp):
        indices = student_out.argmax(dim=-1)
        print('indices', indices.shape)
        pred = student.module.predictor(indices)
        print('pred', pred.shape)
        encoding = (teacher_out[:, 1:] - teacher_out[:, :-1]) / temp
        loss = torch.sum(-F.softmax(encoding, dim=-1) * F.log_softmax(pred[:, :-1] / self.student_temp, dim=-1), dim=-1)
        return loss.mean()

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
