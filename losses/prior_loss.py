__all__ = ['PriorLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class PriorLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, args=None, **kwargs):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.args = args
        self.register_buffer("center", torch.zeros(1, 1, out_dim))
        self.register_buffer("predict_future_center", torch.zeros(1, 1, out_dim))
        self.register_buffer("predict_past_center", torch.zeros(1, 1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc_logits, s_pred_future_logits, indices, s_future_pos_emb, s_alpha_pos = student_output
        t_enc_logits, t_pred_future_logits, indices, t_future_pos_emb, t_alpha_pos = teacher_output

        temp = self.teacher_temp_schedule[epoch]

        # s_enc_proba = F.softmax(s_enc_logits / self.student_temp, dim=-1)
        # t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        # s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
        # t_pred_future_proba = F.softmax((t_pred_future_logits - self.predict_future_center) / temp, dim=-1)

        CE_fe = self.compute_loss_fe(s_pred_future_logits, t_enc_logits, t_alpha_pos, temp)
        CE_ef = self.compute_loss_ef(s_enc_logits, t_pred_future_logits, s_alpha_pos, temp)

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        self.update_centers(t_enc_logits, None, None)
        # time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'entropy': self.entropy(self.center),
                            # 'batch_time_entropy': time_entropy,
                            # 'KL': KL,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_loss_fe(self, s_pred_future_logits, t_enc_logits, t_alpha_pos, temp):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ie in range(1, self.n_crops):  # future encoding
            s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)

            t_alpha_pos_ie = nn.functional.normalize(t_alpha_pos[:, ie:], dim=1, p=2)
            weighted_t_enc_logits = t_alpha_pos_ie * t_enc_logits[:, ie:]
            t_enc_proba = F.softmax((weighted_t_enc_logits.mean(1, keepdim=True) - self.center) / temp, dim=-1)
            # print('t_enc_proba', t_enc_proba.shape)
            # print('s_pred_future_proba', s_pred_future_proba.shape)
            loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    def compute_loss_ef(self, s_enc_logits, t_pred_future_logits, s_alpha_pos, temp):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ie in range(1, self.n_crops):  # future encoding
            t_pred_future_proba = F.softmax((t_pred_future_logits - self.center) / temp, dim=-1)

            s_alpha_pos_ie = nn.functional.normalize(s_alpha_pos[:, ie:], dim=1, p=2)
            weighted_s_enc_logits = s_alpha_pos_ie * s_enc_logits[:, ie:]
            s_enc_proba = F.softmax(weighted_s_enc_logits[:, ie:].mean(1, keepdim=True) / self.student_temp, dim=-1)
            # print('s_enc_proba', s_enc_proba.shape)
            # print('t_pred_future_proba[:, :ie]', t_pred_future_proba.shape)
            loss = -torch.sum(t_pred_future_proba * torch.log(s_enc_proba), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    def dirac_entropy(self, t_enc_logits):
        labels = torch.argmax(t_enc_logits, dim=-1)
        onehot = F.one_hot(labels)
        time_dirac_proba = onehot.float().mean(dim=1)
        dirac_entropy = -torch.sum(time_dirac_proba * torch.log(time_dirac_proba + 1e-8), dim=-1).mean()
        max_entropy = np.log(onehot.size(1))
        dirac_entropy_proportion2max = dirac_entropy / max_entropy
        return dirac_entropy, dirac_entropy_proportion2max

    def time_entropy(self, t_enc_proba):
        time_events_proba = t_enc_proba.mean(1)
        time_entropy = -torch.sum(time_events_proba * torch.log(time_events_proba), dim=-1).mean()
        return time_entropy

    @torch.no_grad()
    def update_centers(self, t_enc_logits, t_pred_future_logits, t_pred_past_logits):
        # update batch centers
        batch_center = self.get_batch_center(t_enc_logits)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        if t_pred_future_logits is not None:
            batch_center_pred_future = self.get_batch_center(t_pred_future_logits)
            self.predict_future_center = self.predict_future_center * self.center_momentum \
                                         + batch_center_pred_future * (1 - self.center_momentum)

        if t_pred_past_logits is not None:
            batch_center_pred_past = self.get_batch_center(t_pred_past_logits)
            self.predict_past_center = self.predict_past_center * self.center_momentum \
                                       + batch_center_pred_past * (1 - self.center_momentum)

    @torch.no_grad()
    def entropy(self, x):
        return -torch.sum(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1), dim=-1).mean()

    def compute_kl(self, conditional):
        marginal_log = F.log_softmax(self.center.detach(), dim=-1).repeat(conditional.size(0), conditional.size(1), 1)
        conditional_log = torch.log(conditional)
        kl = F.kl_div(conditional_log, marginal_log, log_target=True)
        return kl.mean()

    def get_batch_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        b, t, *_ = teacher_output.shape
        batch_center = torch.sum(torch.sum(teacher_output, dim=0, keepdim=True), dim=1,  keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (b * t * dist.get_world_size())
        return batch_center

