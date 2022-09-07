__all__ = ['MemoryLoss']

import torch
import torch.nn.functional as F
from collections import deque

from .te_pp_loss import TEPPLoss


class MemoryLoss(TEPPLoss):
    def init_memory(self, batch_size=None, maxlen=16):
        if self.args.continuous:
            self.memory = deque(maxlen=maxlen)
            self.memory_mask = deque(maxlen=maxlen)
            self.current_video_indices = torch.zeros(batch_size)

    def add_memory(self, values):
        if self.args.continuous:
            print('add_memory values', values.shape)
            self.memory.append(values.detach())
            self.memory_mask.append(torch.ones(self.batch_size))
            print('add_memory self.memory_mask', self.memory_mask)

    def remove_memory(self, video_indices):
        new_video_indices = ~(self.current_video_indices == video_indices)
        print('remove_memory new_video_indices', new_video_indices)
        for idx in video_indices[new_video_indices]:
            print('remove_memory idx')
            for i in range(len(self.memory)):
                self.memory[i][idx] = torch.zeros_like(self.memory[i][idx])
                self.memory_mask[i][idx] = 0

    def retrieve_memory(self, ):
        if self.args.continuous:
            return torch.stack(self.memory, 1), torch.stack(self.memory_mask, 1)

    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None,
                video_indices=None, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc, s_enc_logits, s_indices = student_output
        t_enc, t_enc_logits, t_indices = teacher_output

        temp = self.teacher_temp_schedule[epoch]

        s_enc_proba = F.softmax(s_enc_logits / self.student_temp, dim=-1)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        self.remove_memory(video_indices)
        memory_enc, memory_mask = self.retrieve_memory()
        print('memory', memory_enc.shape)
        print('memory_mask', memory_mask.shape)

        CE_fe = self.compute_loss_fe(memory_enc, t_enc_proba, student, t_indices)
        CE_ef = self.compute_loss_ef(s_enc_proba, memory_enc, teacher, t_indices, temp)

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        self.add_memory(t_enc_logits[:, 0])
        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, memory_enc, t_enc_proba, student, indices):
        print('compute_loss_fe')
        print('memory_enc, t_enc_proba', memory_enc.shape, t_enc_proba.shape)
        s_pred_future = student.module.predictor(memory_enc)
        print('s_pred_future', s_pred_future.shape)
        s_pred_future_logits = student.module.head(s_pred_future)
        print('s_pred_future_logits', s_pred_future_logits.shape)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
        loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
        print('loss', loss.shape, self.memory_mask.shape)
        total_loss = (self.memory_mask * loss).sum() / self.memory_mask.sum()
        return total_loss

    def compute_loss_ef(self, s_enc_proba, memory_enc, teacher, indices, temp):
        print('compute_loss_ef')
        print('memory_enc, s_enc_proba', memory_enc.shape, s_enc_proba.shape)
        t_pred_future = teacher.predictor(memory_enc)
        print('t_pred_future', t_pred_future.shape)
        t_pred_future_logits = teacher.head(t_pred_future)
        print('t_pred_future_logits', t_pred_future_logits.shape)
        t_pred_future_proba = F.softmax((t_pred_future_logits - self.center) / temp, dim=-1)
        loss = -torch.sum(t_pred_future_proba * torch.log(s_enc_proba), dim=-1)
        print('loss', loss.shape, self.memory_mask.shape)
        total_loss = (self.memory_mask * loss).sum() / self.memory_mask.sum()
        return total_loss
