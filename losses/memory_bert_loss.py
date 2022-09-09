__all__ = ['MemoryBertLoss']

import torch
import torch.nn.functional as F
from collections import deque

from .memory_loss import MemoryLoss


class MemoryBertLoss(MemoryLoss):
    def get_mask(self, indices, masking_ratio=0.2):
        rand = torch.rand(indices.shape)
        mask_arr = (rand > masking_ratio).long()
        return mask_arr

    def random_masking(self, x, mask_ratio):
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

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        print('ids_restore', ids_restore.shape)

        x_masked = x * mask
        return x_masked, mask

    def get_pos_indices(self, memory_enc):
        indices = torch.arange(memory_enc.size(1)+1).flip([0]).unsqueeze(0).repeat(memory_enc.size(0), 1).to(
        memory_enc.device)
        return indices

    def compute_loss_fe(self, memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices):
        memory_enc_masked, token_memory_mask = self.random_masking(memory_enc, mask_ratio=self.args.masking_ratio)
        print('token_memory_mask, memory_enc', token_memory_mask.shape, memory_enc.shape)
        x_masked = torch.cat([memory_enc_masked, torch.zeros_like(memory_enc[:, -1:])], 1)
        print('x_masked', x_masked.shape)
        x_token_mask = torch.cat([token_memory_mask, torch.zeros_like(token_memory_mask[:, -1:])], 1)
        print('x_token_mask', x_token_mask.shape)
        s_pred_future = student.module.predictor(x_masked, indices=pos_indices, token_mask=x_token_mask,
                                                 attn_type='single')
        print('s_pred_future', s_pred_future.shape)
        s_pred_future = s_pred_future[:, -1:]
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
        t_pred_future = teacher.predictor(memory_enc, indices=pos_indices)
        t_pred_future_logits = teacher.head(t_pred_future)
        t_pred_future_proba = F.softmax((t_pred_future_logits - self.predict_future_center) / temp, dim=-1)
        s_enc_log = torch.log(s_enc_proba).mean(1, keepdim=True)
        loss = -torch.sum(t_pred_future_proba * s_enc_log, dim=-1)
        mask_sum = memory_mask.sum() + 1e-16
        total_loss = (memory_mask * loss).sum() / mask_sum

        self.update_centers(None, t_pred_future_logits, None)
        return total_loss

