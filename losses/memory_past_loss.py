__all__ = ['MemoryPastLoss']

import torch
import torch.nn.functional as F

from .memory_bert_loss import MemoryBertLoss


class MemoryPastLoss(MemoryBertLoss):
       def compute_loss_fe(self, memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices):
        s_pred_future = student.module.predictor(memory_enc, indices=pos_indices)
        s_pred_future_logits = student.module.head(s_pred_future)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
        mask_sum = memory_mask.sum() + 1e-16
        total_loss = (memory_mask * loss).sum() / mask_sum
        return total_loss
