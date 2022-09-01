__all__ = ['DINOGumbelLoss', 'DINOGumbel2Loss',
           'DINOTopkLoss', 'DINOGumbel3Loss',
           'DINORandomChoiceLoss']

import torch
import torch.nn.functional as F

from .dino_loss import DINOLoss


class DINOGumbelLoss(DINOLoss):
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
        teacher_out = F.gumbel_softmax((teacher_output - self.center) / temp, dim=-1, hard=True)
        teacher_out = teacher_out.detach().chunk(self.global_crops)

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                # loss = F.nll_loss(F.log_softmax(student_out[v], dim=-1), q.argmax(dim=-1))
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


class DINOGumbel2Loss(DINOLoss):
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
        teacher_output_norm = (teacher_output - self.center) / temp
        teacher_out = F.gumbel_softmax(teacher_output_norm, dim=-1, hard=True)
        for _ in range(99):
            teacher_out += F.gumbel_softmax(teacher_output_norm, dim=-1, hard=True)
        teacher_out /= 100
        teacher_out = teacher_out.detach().chunk(self.global_crops)

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                # loss = F.nll_loss(F.log_softmax(student_out[v], dim=-1), q.argmax(dim=-1))
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


class DINOTopkLoss(DINOLoss):
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
        teacher_out_proba = F.softmax((teacher_output - self.center) / temp, dim=-1)
        topK = 100
        proba, teacher_out = teacher_out_proba.topk(k=topK, dim=-1)
        proba = proba / proba.sum(-1, keepdims=True)

        teacher_out = teacher_out.detach().chunk(self.global_crops)
        proba = proba.detach().chunk(self.global_crops)

        for iq, (q, p) in enumerate(zip(teacher_out, proba)):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = 0
                # b_idx = torch.arange(teacher_out.size(0)).unsqueeze(1).repeat(1, topK)
                # p = teacher_out_proba[b_idx, teacher_out]
                for itk in range(topK):
                    loss += p[:, itk]*F.nll_loss(F.log_softmax(student_out[v], dim=-1), q[:, itk])
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


from torch.distributions.categorical import Categorical
from torch import distributions as torchd


class DINOGumbel3Loss(DINOLoss):
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
        teacher_out_proba = F.softmax((teacher_output - self.center) / temp, dim=-1)
        # dist = torchd.independent.Independent(Categorical(probs=teacher_out_proba), 0)
        dist = Categorical(probs=teacher_out_proba)
        topK = 10
        teacher_out = dist.sample(sample_shape=(topK, )).T
        b_idx = torch.arange(teacher_out.size(0)).unsqueeze(1).repeat(1, topK)
        proba = teacher_out_proba[b_idx, teacher_out]
        proba = proba / proba.sum(-1, keepdims=True)

        teacher_out = teacher_out.detach().chunk(self.global_crops)
        proba = proba.detach().chunk(self.global_crops)

        for iq, (q, p) in enumerate(zip(teacher_out, proba)):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = 0
                for itk in range(topK):
                    loss += p[:, itk]*F.nll_loss(F.log_softmax(student_out[v], dim=-1), q[:, itk])
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


class DINORandomChoiceLoss(DINOLoss):
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
        teacher_out_proba = F.softmax((teacher_output - self.center) / temp, dim=-1)
        topK = 10
        teacher_out = teacher_out_proba.multinomial(num_samples=topK, replacement=False)
        b_idx = torch.arange(teacher_out.size(0)).unsqueeze(1).repeat(1, topK)
        proba = teacher_out_proba[b_idx, teacher_out]
        proba = proba / proba.sum(-1, keepdims=True)

        teacher_out = teacher_out.detach().chunk(self.global_crops)
        proba = proba.detach().chunk(self.global_crops)

        for iq, (q, p) in enumerate(zip(teacher_out, proba)):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = 0
                for itk in range(topK):
                    loss += p[:, itk]*F.nll_loss(F.log_softmax(student_out[v], dim=-1), q[:, itk])
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