__all__ = ['MemoryLoss']

import torch
import torch.nn.functional as F
from collections import deque

from .te_pp_loss import TEPPLoss


class MemoryLoss(TEPPLoss):
    def init_memory(self, batch_size=None, **kwargs):
        self.memory = deque(maxlen=self.args.maxlen)
        self.memory_mask = deque(maxlen=self.args.maxlen)
        self.current_video_indices = -torch.ones(batch_size)

    def add_memory(self, values):
        # print('add_memory values', values.shape)
        self.memory.append(values.detach())
        self.memory_mask.append(torch.ones(self.batch_size).to(values.device))
        # print('add_memory self.memory_mask', self.memory_mask)

    def remove_memory(self, video_indices):
        # print('self.current_video_indices', self.current_video_indices)
        # print('video_indices', video_indices)
        new_video_indices = ~(self.current_video_indices == video_indices)
        # print('new_video_indices', new_video_indices)
        self.current_video_indices = video_indices
        # print('before remove_memory self.memory_mask', self.memory_mask)
        for idx in torch.arange(self.batch_size)[new_video_indices]:
            # print('remove_memory idx', idx)
            for i in range(len(self.memory)):
                self.memory[i][idx] = torch.zeros_like(self.memory[i][idx])
                self.memory_mask[i][idx] = 0
        # print('remove_memory self.memory_mask', self.memory_mask)

    def retrieve_memory(self, ):
        return torch.stack(list(self.memory), 1), torch.stack(list(self.memory_mask), 1)

    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None,
                video_indices=None, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        s_enc, s_enc_logits, s_indices = student_output
        t_enc, t_enc_logits, t_indices = teacher_output

        temp = self.teacher_temp_schedule[epoch]

        if self.args.teacher_enc_pred_head:
            s_enc_logits = teacher.head(s_enc)

        s_enc_proba = F.softmax(s_enc_logits / self.student_temp, dim=-1)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        if not self.memory:
            print('add first memory!') #ToDo: fix this trick
            self.add_memory(t_enc[:, 0])
        self.remove_memory(video_indices)
        memory_enc, memory_mask = self.retrieve_memory()

        pos_indices = self.get_pos_indices(memory_enc)
        CE_fe = self.compute_loss_fe(memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices) if self.args.CE_fe_c else 0.
        CE_ef = self.compute_loss_ef(s_enc_proba, memory_enc, memory_mask, student, teacher, pos_indices, temp) if self.args.CE_ef_c else 0.
        CE_ee = self.dino_loss(t_enc_proba, s_enc_proba) if self.args.CE_ee_c else 0.

        memory_size = memory_mask.sum(-1).mean()
        if self.args.memory_balance_loss:
            total = memory_size + (self.n_crops - 1)
            CE_ef_c = (self.n_crops - 1) / total
            CE_ee_c = memory_size / total
            total_loss = self.args.CE_fe_c * CE_fe + CE_ef_c * CE_ef + CE_ee_c * CE_ee
        else:
            total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef + self.args.CE_ee_c * CE_ee

        self.add_memory(t_enc[:, 0])
        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss / (self.args.CE_fe_c + self.args.CE_ef_c + self.args.CE_ee_c),
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'CE_ee': CE_ee,
                            'memory_size': memory_size,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def get_pos_indices(self, memory_enc):
        indices = torch.arange(memory_enc.size(1)).flip([0]).unsqueeze(0).repeat(memory_enc.size(0), 1).to(
        memory_enc.device)
        return indices

    def compute_loss_fe(self, memory_enc, memory_mask, t_enc_proba, student, teacher, pos_indices):
        # print('compute_loss_fe')
        # print('memory_enc, t_enc_proba', memory_enc.shape, t_enc_proba.shape)
        s_pred_future = student.module.predictor(memory_enc, indices=pos_indices)
        # print('s_pred_future', s_pred_future.shape)
        if self.args.teacher_pred_head:
            s_pred_future_logits = teacher.head(s_pred_future)
        else:
            s_pred_future_logits = student.module.head(s_pred_future)
        # print('s_pred_future_logits', s_pred_future_logits.shape)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
        # print('t_enc_proba', t_enc_proba.shape)
        t_enc_proba = t_enc_proba.mean(dim=1, keepdim=True)
        # print('t_enc_proba 2', t_enc_proba.shape)

        if self.args.memory_offset:
            s_pred_future_proba = s_pred_future_proba[:, self.args.memory_offset:]
            memory_mask = memory_mask[:, self.args.memory_offset:]

        loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
        # print('loss', loss.shape, memory_mask.shape)
        mask_sum = memory_mask.sum() + 1e-16
        # print('memory_mask', memory_mask)
        # print('mask_sum', mask_sum)
        total_loss = (memory_mask * loss).sum() / mask_sum
        return total_loss

    def compute_loss_ef(self, s_enc_proba, memory_enc, memory_mask, student, teacher, pos_indices, temp):
        # print('compute_loss_ef')
        # print('memory_enc, s_enc_proba', memory_enc.shape, s_enc_proba.shape)
        t_pred_future = teacher.predictor(memory_enc, indices=pos_indices)
        # print('t_pred_future', t_pred_future.shape)
        t_pred_future_logits = teacher.head(t_pred_future)
        # print('t_pred_future_logits', t_pred_future_logits.shape)
        t_pred_future_proba = F.softmax((t_pred_future_logits - self.predict_future_center) / temp, dim=-1)
        # print('s_enc_proba', s_enc_proba.shape)
        s_enc_log = torch.log(s_enc_proba).mean(1, keepdim=True)
        # print('s_enc_log', s_enc_log.shape)
        if self.args.memory_offset:
            s_enc_log = s_enc_log[:, self.args.memory_offset:]
            memory_mask = memory_mask[:, self.args.memory_offset:]

        loss = -torch.sum(t_pred_future_proba * s_enc_log, dim=-1)
        # print('loss', loss.shape, memory_mask.shape)
        mask_sum = memory_mask.sum() + 1e-16
        # print('mask_sum', mask_sum)
        total_loss = (memory_mask * loss).sum() / mask_sum

        self.update_centers(None, t_pred_future_logits, None)
        return total_loss

    def dino_loss(self, t_enc_proba, s_enc_proba):
        total_loss = 0
        n_loss_terms = 0
        for iq in range(self.n_global_views):
            for v in range(self.n_crops):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-t_enc_proba[:, iq] * torch.log(s_enc_proba[:, v]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        n_loss_terms = max(1, n_loss_terms)
        total_loss /= n_loss_terms
        return total_loss
