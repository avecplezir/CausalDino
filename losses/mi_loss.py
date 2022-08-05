__all__ = ['MILoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class MILoss(nn.Module):
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

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        CE = 0
        n_loss_terms = 0

        student_out = student_output
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output, dim=-1)
        student_proba = F.softmax(student_output, dim=1)
        batch_center = self.update_center_get_batch_center(student_proba)
        teacher_out = teacher_out.chunk(self.global_crops)

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                CE += loss.mean()
                n_loss_terms += 1
        entropy = torch.sum(batch_center * torch.log(self.center), dim=-1)
        entropy_plus = entropy + batch_center.sum()
        CE /= n_loss_terms
        total_loss = CE + entropy_plus

        true_entropy = torch.sum(self.center * torch.log(self.center), dim=-1)
        return total_loss, {'CE': CE, 'entropy': entropy, 'true_entropy': true_entropy, 'mean': batch_center.sum()}

    @torch.no_grad()
    def update_center_get_batch_center(self, teacher_output):
        """
        Update center used for teacher output.
        """

        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center.detach() * (1 - self.center_momentum)

        return batch_center
