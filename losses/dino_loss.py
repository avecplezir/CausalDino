__all__ = ['DINOLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, two_token=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.two_token = two_token
        if self.two_token:
            self.n_crops = 4
            self.global_crops = 2
            self.register_buffer("center", torch.zeros(2, out_dim))
        else:
            self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        n_loss_terms = 0
        if self.two_token:
            student_out = [x / self.student_temp for x in student_output]
            student_out = [x.chunk(self.n_crops) for x in student_out]

            # teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = [F.softmax((x - self.center[idx]) / temp, dim=-1) for idx, x in enumerate(teacher_output)]
            teacher_out = [x.detach().chunk(self.global_crops) for x in teacher_out]

            for iv in range(len(student_out[0])):
                if iv < 2:
                    q = teacher_out[0][0]
                    v = student_out[0][iv]
                    loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                else:
                    q = teacher_out[1][1]
                    v = student_out[1][iv]
                    loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        else:
            student_out = student_output / self.student_temp
            student_out = student_out.chunk(self.n_crops)

            # teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
            teacher_out = teacher_out.detach().chunk(self.global_crops)

            for iq, q in enumerate(teacher_out):
                for v in range(len(student_out)):
                    if v == iq:
                        # we skip cases where student and teacher operate on the same view
                        continue
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    total_loss += loss.mean()
                    n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        if isinstance(teacher_output, (tuple, list)):
            batch_center = [torch.sum(x, dim=0, keepdim=True) for x in teacher_output]
            dist.all_reduce(batch_center[0])
            dist.all_reduce(batch_center[1])
            batch_center = [x / (len(teacher_output[0]) * dist.get_world_size()) for x in batch_center]
            self.center[0, :] = self.center[0, :] * self.center_momentum + batch_center[0] * (1 - self.center_momentum)
            self.center[1, :] = self.center[1, :] * self.center_momentum + batch_center[1] * (1 - self.center_momentum)
        else:
            batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

            # ema update
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
