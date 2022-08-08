__all__ = ['GPTLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class GPTLoss(nn.Module):
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
        student_out = student_output.chunk(self.n_crops)
        student_out = torch.stack(student_out, 1)
        print('student_out', student_out.shape)
        student_out = student.module.predictor(student_out) / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_output = teacher.predictor.last_layer(teacher_output)
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.n_crops)
        teacher_out = torch.stack(teacher_out, 1)
        print('teacher_out', teacher_out.shape)

        loss = torch.sum(-teacher_out[:, 1:] * F.log_softmax(student_out[:, :-1], dim=-1), dim=-1)
        print('loss', loss.shape)
        total_loss = loss.mean()

        batch_center = self.update_center(teacher_output)

        true_entropy = torch.sum(F.softmax(self.center, dim=-1) * F.log_softmax(self.center), dim=-1)
        entropy = torch.sum(F.softmax(batch_center, dim=-1) * F.log_softmax(self.center), dim=-1)

        return total_loss, {'CE': total_loss, 'entropy': entropy, 'true_entropy': true_entropy}

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
