__all__ = ['MemoryBertLoss']

import torch
import torch.nn.functional as F
from collections import deque

from .memory_loss import MemoryLoss


class MemoryBertLoss(MemoryLoss):
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

        memory_enc = torch.cat([memory_enc, s_enc[:, :1]], 1)
        memory_mask = torch.cat([memory_mask, torch.ones_like(memory_mask[:, -1:])])
        pos_indices = self.get_pos_indices(memory_enc)

        temp = self.teacher_temp_schedule[epoch]

        t_enc_logits = teacher.head(memory_enc)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        CE_fe = self.compute_loss_fe(memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices)

        total_loss = CE_fe
        memory_size = memory_mask.sum(-1).mean()

        self.add_memory(t_enc[:, 0])
        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss / (self.args.CE_fe_c + self.args.CE_ef_c + self.args.CE_ee_c),
                            'CE_fe': CE_fe,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def random_masking(self, x, mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise = noise * mask

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1, descending=True)  # descend: large is keep, small is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        print('ids_restore', ids_restore.shape)

        x_masked = x * mask
        return x_masked, mask

    def compute_loss_fe(self, memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices):
        print('memory_enc', memory_enc.shape)
        print('memory_mask', memory_mask.shape)
        print('t_enc_proba', t_enc_proba.shape)
        print('pos_indices', pos_indices.shape)
        memory_enc_masked, token_memory_mask = self.random_masking(memory_enc, memory_mask,
                                                                   mask_ratio=self.args.masking_ratio)
        print('token_memory_mask, memory_enc_masked', token_memory_mask.shape, memory_enc_masked.shape)
        s_pred_future = student.module.predictor(memory_enc_masked, indices=pos_indices, token_mask=token_memory_mask,
                                                 attn_type='all')
        print('s_pred_future', s_pred_future.shape)
        s_pred_future_logits = student.module.head(s_pred_future)
        print('s_pred_future_logits', s_pred_future_logits.shape)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
        print('loss', loss.shape)

        mask = token_memory_mask * memory_mask
        print('token_memory_mask', token_memory_mask)
        print('memory_mask', memory_mask)
        print('mask', mask)
        mask_sum = mask.sum() + 1e-16
        total_loss = (mask * loss).sum() / mask_sum

        return total_loss


