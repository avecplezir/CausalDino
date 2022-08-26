__all__ = ['DINOMILoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class DINOMILoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, args=None, **kwargs):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.args = args
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc_logits = torch.stack(student_output.chunk(self.n_crops), 1)
        s_proba = F.softmax(s_enc_logits, dim=-1)
        s_log = torch.log(s_proba)
        s_marginal = s_proba.mean(0).mean(0)

        CE = -torch.sum(s_proba[:, 0] * s_log[:, 1], dim=-1) \
             + -torch.sum(s_proba[:, 1] * s_log[:, 0], dim=-1)
        CE = (CE / 2).mean()

        entropy = torch.sum(s_marginal * torch.log(s_marginal), dim=-1)
        total_loss = CE + self.args.coef_entropy * entropy

        time_events_proba = F.softmax(s_enc_logits, dim=-1).mean(1)
        time_entropy = -torch.sum(time_events_proba * torch.log(time_events_proba), dim=-1).mean()

        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(s_enc_logits)

        return total_loss, {'CE': CE,
                            'entropy': entropy,
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def compute_kl(self, conditional):
        marginal_log = F.log_softmax(self.center.detach(), dim=-1).repeat(conditional.size(0), conditional.size(1), 1)
        conditional_log = torch.log(conditional)
        kl = F.kl_div(conditional_log, marginal_log, log_target=True)
        return kl.mean()

    def dirac_entropy(self, t_enc_logits):
        labels = torch.argmax(t_enc_logits, dim=-1)
        onehot = F.one_hot(labels)
        time_dirac_proba = onehot.float().mean(dim=1)
        dirac_entropy = -torch.sum(time_dirac_proba * torch.log(time_dirac_proba + 1e-8), dim=-1).mean()
        max_entropy = np.log(onehot.size(1))
        dirac_entropy_proportion2max = dirac_entropy / max_entropy
        return dirac_entropy, dirac_entropy_proportion2max
