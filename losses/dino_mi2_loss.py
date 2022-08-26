__all__ = ['DINOMI2Loss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class DINOMI2Loss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, **kwargs):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
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
        CE = 0
        n_loss_terms = 0
        student_out = F.softmax(student_output, dim=1)
        student_out = student_out.chunk(self.n_crops)
        teacher_out = student_out[:2]

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                s = torch.log(student_out[v]).mean(0, keepdim=True)
                q = q.mean(0, keepdim=True)
                loss = torch.sum(-q * s, dim=-1)
                CE += loss.mean()
                n_loss_terms += 1
        CE /= n_loss_terms

        q = teacher_out.mean(0, keepdim=True)
        entropy = torch.sum(q * torch.log(q), dim=-1)

        total_loss = CE + entropy.mean()

        s_enc_logits = torch.stack(student_output.chunk(self.n_crops), 1)
        time_events_proba = F.softmax(s_enc_logits, dim=-1).mean(1)
        time_entropy = -torch.sum(time_events_proba * torch.log(time_events_proba), dim=-1).mean()
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(s_enc_logits)

        return total_loss, {'CE': CE,
                            'entropy': entropy,
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def get_batch_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        b, *_ = teacher_output.shape
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (b * dist.get_world_size())
        return batch_center

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

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return batch_center
