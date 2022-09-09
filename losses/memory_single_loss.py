__all__ = ['MemorySingleLoss']

import torch
import torch.nn.functional as F
from collections import deque

from .memory_loss import MemoryLoss


class MemorySingleLoss(MemoryLoss):
    def random_masking(self, x, memory_mask, pos_indices, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        pos_indices_masked = torch.gather(pos_indices, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        memory_mask_masked = torch.gather(memory_mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        return x_masked, memory_mask_masked, pos_indices_masked

    def compute_loss_fe(self, memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices):
        print('memory_mask', memory_mask.shape)
        memory_enc_masked, memory_mask, pos_indices_masked = self.random_masking(memory_enc, memory_mask, pos_indices,
                                                                    mask_ratio=self.args.masking_ratio)
        print('token_memory_mask, memory_enc', pos_indices_masked.shape, memory_enc_masked.shape)
        s_pred_future = student.module.predictor(memory_enc_masked, indices=pos_indices_masked)
        print('s_pred_future', s_pred_future.shape)
        if self.args.teacher_pred_head:
            s_pred_future_logits = teacher.head(s_pred_future)
        else:
            s_pred_future_logits = student.module.head(s_pred_future)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
        t_enc_proba = t_enc_proba.mean(dim=1, keepdim=True)
        loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
        print('loss', loss.shape)
        total_loss = loss.mean()

        return total_loss

    def compute_loss_ef(self, s_enc_proba, memory_enc, memory_mask, student, teacher, pos_indices, temp):
        print('memory_mask', memory_mask.shape)
        memory_enc_masked, memory_mask, pos_indices_masked = self.random_masking(memory_enc, memory_mask, pos_indices,
                                                                    mask_ratio=self.args.masking_ratio)
        print('token_memory_mask, memory_enc', pos_indices_masked.shape, memory_enc_masked.shape)
        t_pred_future = teacher.predictor(memory_enc_masked, indices=pos_indices_masked)
        print('t_pred_future', t_pred_future.shape)
        t_pred_future_logits = teacher.head(t_pred_future)
        t_pred_future_proba = F.softmax((t_pred_future_logits - self.predict_future_center) / temp, dim=-1)
        s_enc_log = torch.log(s_enc_proba).mean(1, keepdim=True)
        loss = -torch.sum(t_pred_future_proba * s_enc_log, dim=-1)
        mask_sum = memory_mask.sum() + 1e-16
        total_loss = (memory_mask * loss).sum() / mask_sum

        self.update_centers(None, t_pred_future_logits, None)
        return total_loss

