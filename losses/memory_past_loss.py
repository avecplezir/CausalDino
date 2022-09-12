__all__ = ['MemoryPastLoss']

import torch
import torch.nn.functional as F
from collections import deque

from .memory_loss import MemoryLoss


class MemoryPastLoss(MemoryLoss):
    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None,
                video_indices=None, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        s_enc, _, s_indices = student_output
        t_enc, _, t_indices = teacher_output

        if not self.memory:
            print('add first memory!') #ToDo: fix this trick
            self.add_memory(t_enc[:, 0])
        self.remove_memory(video_indices)
        memory_enc, memory_mask = self.retrieve_memory()

        s_memory_enc = torch.cat([memory_enc, s_enc[:, :1]], 1)
        t_memory_enc = torch.cat([memory_enc, t_enc[:, :1]], 1)
        memory_mask = torch.cat([memory_mask, torch.ones_like(memory_mask[:, -1:])], 1)
        pos_indices = self.get_pos_indices(t_memory_enc)

        temp = self.teacher_temp_schedule[epoch]

        t_enc_logits = teacher.head(t_memory_enc)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        CE_fe = self.compute_loss_fe(s_memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices)

        total_loss = CE_fe
        memory_size = memory_mask.sum(-1).mean()

        self.add_memory(t_enc[:, 0])
        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices):
        s_pred_future = student.module.predictor(memory_enc, indices=pos_indices)
        if self.args.teacher_pred_head:
            s_pred_future_logits = teacher.head(s_pred_future)
        else:
            s_pred_future_logits = student.module.head(s_pred_future)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
        t_enc_proba = t_enc_proba.mean(dim=1, keepdim=True)

        if self.args.memory_offset:
            s_pred_future_proba = s_pred_future_proba[:, -self.args.memory_offset:]
            memory_mask = memory_mask[:, -self.args.memory_offset:]

        loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
        mask_sum = memory_mask.sum() + 1e-16
        total_loss = (memory_mask * loss).sum() / mask_sum
        return total_loss
