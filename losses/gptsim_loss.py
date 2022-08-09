__all__ = ['GPTSimLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class GPTSimLoss(nn.Module):
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
        self.register_buffer("prediction_center", torch.zeros(1, out_dim))
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

        teacher_out = teacher_output.detach().chunk(self.n_crops)
        teacher_out = torch.stack(teacher_out, 1)
        temp = self.teacher_temp_schedule[epoch]

        # First direction
        student_logits_predictions = student.module.predictor(student_out) / self.student_temp
        # teacher centering and sharpening
        teacher_logits = teacher.predictor.last_layer(teacher_out)
        labels = F.softmax((teacher_logits - self.center) / temp, dim=-1)
        CE_fp = torch.sum(-labels[:, 1:] * F.log_softmax(student_logits_predictions[:, :-1], dim=-1), dim=-1)

        # Second direction
        student_logits = student.module.predictor.last_layer(student_out) / self.student_temp
        # teacher centering and sharpening
        teacher_logits_predictions = teacher.predictor(teacher_out)
        labels_predictions = F.softmax((teacher_logits_predictions - self.prediction_center) / temp, dim=-1)
        CE_pf = torch.sum(-labels_predictions[:, 1:] * F.log_softmax(student_logits[:, :-1], dim=-1), dim=-1)

        CE_fp = CE_fp.mean()
        CE_pf = CE_pf.mean()
        total_loss = 0.8*CE_fp + 0.2*CE_pf

        batch_center = self.get_batch_center(teacher_logits)
        self.update_center(batch_center)
        true_entropy = torch.sum(F.softmax(self.center, dim=-1) * F.log_softmax(self.center), dim=-1)
        entropy = torch.sum(F.softmax(batch_center, dim=-1) * F.log_softmax(self.center), dim=-1)

        batch_center_prediction = self.get_batch_center(teacher_logits_predictions)
        self.update_prediction_center(batch_center_prediction)
        true_entropy_prediction = torch.sum(F.softmax(self.center, dim=-1) * F.log_softmax(self.center), dim=-1)
        entropy_prediction = torch.sum(F.softmax(batch_center_prediction, dim=-1) * F.log_softmax(self.center), dim=-1)

        return total_loss, {'CE': total_loss, 'CE_fp': CE_fp, 'CE_pf': CE_pf,
                            'entropy': entropy, 'true_entropy': true_entropy,
                            'true_entropy_prediction': true_entropy_prediction,
                            'entropy_prediction': entropy_prediction}

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

    @torch.no_grad()
    def update_center(self, batch_center):
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return batch_center

    @torch.no_grad()
    def update_prediction_center(self, batch_center):
        """
        Update center used for teacher output.
        """
        # ema update
        self.prediction_center = self.prediction_center * self.center_momentum + batch_center * (1 - self.center_momentum)
