__all__ = ['DINOLoss', 'DINOOneStudentLoss', 'DINOOneTeacherLoss', 'DINOOneStudent2Loss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, n_global_views=2, **kwargs):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.n_global_views = n_global_views
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
        total_loss = 0
        n_loss_terms = 0
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.global_crops)

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        s_enc_logits = torch.stack(student_out, 1)
        time_events_proba = F.softmax(s_enc_logits, dim=-1)
        time_entropy = self.time_entropy(time_events_proba)

        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(s_enc_logits)

        return total_loss, {'CE': total_loss,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def dirac_entropy(self, t_enc_logits):
        labels = torch.argmax(t_enc_logits, dim=-1)
        onehot = F.one_hot(labels)
        time_dirac_proba = onehot.float().mean(dim=1)
        dirac_entropy = -torch.sum(time_dirac_proba * torch.log(time_dirac_proba + 1e-8), dim=-1).mean()
        max_entropy = max(np.log(onehot.size(1)), 1)
        dirac_entropy_proportion2max = dirac_entropy / max_entropy
        return dirac_entropy, dirac_entropy_proportion2max

    @torch.no_grad()
    def entropy(self, x):
        return -torch.sum(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1), dim=-1).mean()

    @torch.no_grad()
    def time_entropy(self, t_enc_proba):
        time_events_proba = t_enc_proba.mean(1)
        time_entropy = -torch.sum(time_events_proba * torch.log(time_events_proba), dim=-1).mean()
        return time_entropy

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


class DINOOneStudentLoss(DINOLoss):
    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        n_loss_terms = 0
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.n_global_views)

        for iq, q in enumerate(teacher_out):
            v = 0
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        s_enc_logits = torch.stack(student_out, 1)
        time_events_proba = F.softmax(s_enc_logits, dim=-1)
        time_entropy = self.time_entropy(time_events_proba)

        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(s_enc_logits)

        return total_loss, {'CE': total_loss,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }


class DINOOneTeacherLoss(DINOLoss):
    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        n_loss_terms = 0
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.n_global_views)

        for iq, q in enumerate(student_out):
            v = 0
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = torch.sum(-teacher_out[v] * F.log_softmax(q, dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        s_enc_logits = torch.stack(student_out, 1)
        time_events_proba = F.softmax(s_enc_logits, dim=-1)
        time_entropy = self.time_entropy(time_events_proba)

        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(s_enc_logits)

        return total_loss, {'CE': total_loss,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }


from .feature_loss import FeatureLoss


class DINOOneStudent2Loss(FeatureLoss):
    def forward(self, student_output: tuple, teacher_output: tuple, epoch: int, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc_logits, = student_output
        t_enc_logits, = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        CE_fe = self.dino_loss(s_enc_logits, t_enc_logits, temp)

        total_loss = CE_fe

        self.update_centers(t_enc_logits[:, -self.args.n_global_views:], None)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'entropy': self.entropy(self.center),
                            'predict_entropy': self.entropy(self.predict_center),
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def dino_loss(self, s_enc_logits, t_enc_logits, temp):
        s_enc_log = F.log_softmax(s_enc_logits / self.student_temp, dim=-1)
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        assert s_enc_log.size(1) == 1, 'support only one global view!'
        loss = -torch.sum(t_enc_proba * s_enc_log, dim=-1)
        return loss.mean()
