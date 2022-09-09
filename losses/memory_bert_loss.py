__all__ = ['MemoryLoss']

import torch
import torch.nn.functional as F
from collections import deque

from .memory_loss import MemoryLoss


class MemoryBertLoss(MemoryLoss):
    def get_mask(self, indices, masking_ratio=0.2):
        rand = torch.rand(indices.shape)
        mask_arr = rand > masking_ratio
        return mask_arr

    def compute_loss_fe(self, memory_enc, memory_mask, t_enc_proba, student, teacher, indices):
        token_mask = self.get_mask(indices, masking_ratio=self.args.masking_ratio)
        # print('compute_loss_fe')
        # print('memory_enc, t_enc_proba', memory_enc.shape, t_enc_proba.shape)
        s_pred_future = student.module.predictor(memory_enc, indices=indices, token_mask=token_mask)
        # print('s_pred_future', s_pred_future.shape)
        if self.args.teacher_pred_head:
            print('teacher_pred_head!')
            s_pred_future_logits = teacher.head(s_pred_future)
        else:
            s_pred_future_logits = student.module.head(s_pred_future)
        # print('s_pred_future_logits', s_pred_future_logits.shape)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
        # print('t_enc_proba', t_enc_proba.shape)
        t_enc_proba = t_enc_proba.mean(dim=1, keepdim=True)
        # print('t_enc_proba 2', t_enc_proba.shape)
        loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
        # print('loss', loss.shape, memory_mask.shape)
        mask_sum = memory_mask.sum() + 1e-16
        # print('memory_mask', memory_mask)
        # print('mask_sum', mask_sum)
        total_loss = (memory_mask * loss).sum() / mask_sum
        return total_loss

    def compute_loss_ef(self, s_enc_proba, memory_enc, memory_mask, student, teacher, indices, temp):
        token_mask = None
        # print('compute_loss_ef')
        # print('memory_enc, s_enc_proba', memory_enc.shape, s_enc_proba.shape)
        t_pred_future = teacher.predictor(memory_enc, indices=indices, token_mask=token_mask)
        # print('t_pred_future', t_pred_future.shape)
        t_pred_future_logits = teacher.head(t_pred_future)
        # print('t_pred_future_logits', t_pred_future_logits.shape)
        t_pred_future_proba = F.softmax((t_pred_future_logits - self.center) / temp, dim=-1)
        # print('s_enc_proba', s_enc_proba.shape)
        s_enc_log = torch.log(s_enc_proba).mean(1, keepdim=True)
        # print('s_enc_log', s_enc_log.shape)
        loss = -torch.sum(t_pred_future_proba * s_enc_log, dim=-1)
        # print('loss', loss.shape, memory_mask.shape)
        mask_sum = memory_mask.sum() + 1e-16
        # print('mask_sum', mask_sum)
        total_loss = (memory_mask * loss).sum() / mask_sum
        return total_loss
