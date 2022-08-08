__all__ = ['PredLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class PredLoss(nn.Module):
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

    def forward(self, student_output, teacher_output, epoch, student=None, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        n_loss_terms = 0
        student_out = F.softmax(student_output / self.student_temp, dim=-1)
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.global_crops)

        pred = student.module.predictor.get_all()
        for iq, q in enumerate(teacher_out):
            for v, s in enumerate(student_out):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                p_sq = s.unsqueeze(2)*q.unsqueeze(1)
                loss = -torch.sum(torch.sum(p_sq * F.log_softmax(pred, dim=-1), dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        batch_center = self.update_center(teacher_output)

        true_entropy = torch.sum(F.softmax(self.center, dim=-1) * F.log_softmax(self.center), dim=-1)
        entropy = torch.sum(F.softmax(batch_center, dim=-1) * F.log_softmax(self.center), dim=-1)

        return total_loss, {'CE': total_loss, 'entropy': entropy, 'true_entropy': true_entropy}

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

        return batch_center
