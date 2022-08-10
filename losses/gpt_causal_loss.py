__all__ = ['GPTCausalLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class GPTCausalLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, argmax=False, weight_inv=True, **kwargs):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.argmax = argmax
        self.weight_inv = weight_inv
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("predict_future_center", torch.zeros(1, out_dim))
        self.register_buffer("predict_past_center", torch.zeros(1, out_dim))
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

        s_enc_logits, s_pred_future_logits, s_pred_past_logits = student_output
        t_enc_logits, t_pred_future_logits, t_pred_past_logits = teacher_output

        temp = self.teacher_temp_schedule[epoch]

        # Student # Encoding # Future Prediction # Past Prediction
        s_enc_proba = F.softmax(s_enc_logits / self.student_temp, dim=-1)[:, 1:-1]
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)[:, :-2]
        s_pred_past_proba = F.softmax(s_pred_past_logits / self.student_temp, dim=-1)[:, 2:]

        # Teacher # Encoding # Future Prediction # Past Prediction
        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)[:, 1:-1]
        t_pred_future_proba = F.softmax((t_pred_future_logits - self.predict_future_center) / temp, dim=-1)[:, :-2]
        t_pred_past_proba = F.softmax((t_pred_past_logits - self.predict_past_center) / temp, dim=-1)[:, 2:]

        if self.argmax:
            max_index = t_enc_proba.argmax(dim=-1)
            b, t, emb = t_pred_past_proba.shape
            max_index = max_index.reshape(b*t)
            t_pred_past_proba_weight = t_pred_past_proba.reshape(b*t, emb)
            t_pred_past_proba_weight = t_pred_past_proba_weight[torch.arange(b*t), max_index]
            t_pred_past_proba_weight = t_pred_past_proba_weight.reshape(b, t, 1)
            t_pred_future_proba_weight = t_pred_future_proba.reshape(b*t, emb)
            t_pred_future_proba_weight = t_pred_future_proba_weight[torch.arange(b * t), max_index]
            t_pred_future_proba_weight = t_pred_future_proba_weight.reshape(b, t, 1)
        else:
            t_pred_past_proba_weight = torch.sum(t_enc_proba * t_pred_past_proba, dim=-1, keepdim=True)
            t_pred_future_proba_weight = torch.sum(t_enc_proba * t_pred_future_proba, dim=-1, keepdim=True)

        # Losses
        # allign future student prediction with teacher encoding (weighted with past teacher prediction)
        CE_ef = self.compute_loss(s_pred_future_proba, t_enc_proba, t_pred_past_proba_weight)
        # allign future teacher prediction with student encoding (weighted with past teacher prediction)
        CE_fe = self.compute_loss(s_enc_proba, t_pred_future_proba, t_pred_past_proba_weight)
        # allign past student prediction with teacher encoding (weighted with future teacher prediction)
        CE_ep = self.compute_loss(s_pred_past_proba, t_enc_proba, t_pred_future_proba_weight)
        # allign past teacher prediction with student encoding (weighted with future teacher prediction)
        CE_pe = self.compute_loss(s_enc_proba, t_pred_past_proba, t_pred_future_proba_weight)

        total_loss = 0.45*CE_ef + 0.05*CE_fe + 0.45*CE_ep + 0.05*CE_pe
        entropies = self.update_centers(t_enc_logits, t_pred_future_logits, t_pred_past_logits)

        return total_loss, {'CE': total_loss, 'CE_ef': CE_ef, 'CE_fe': CE_fe,
                            'CE_ep': CE_ep, 'CE_pe': CE_pe, **entropies}

    def compute_loss(self, prediction, labels, inverse):
        total_loss = 0
        n_loss_terms = 0
        minimum = 1e-4 * torch.ones_like(inverse[:, 0])
        for ip, p in enumerate(prediction.chunk(self.n_crops-2, dim=1)):
            for il in range(ip + 1, self.n_crops-2):
                if self.weight_inv:
                    inv = torch.max(minimum, 1 - inverse[:, il])
                else:
                    inv = 1
                loss = -torch.sum(labels[:, il] * torch.log(p) / inv, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        return total_loss

    @torch.no_grad()
    def entropy(self, x):
        return torch.sum(F.softmax(x, dim=-1) * F.log_softmax(x), dim=-1)

    @torch.no_grad()
    def approx_entropy(self, x, p):
        return torch.sum(F.softmax(p, dim=-1) * F.log_softmax(x), dim=-1)

    @torch.no_grad()
    def update_centers(self, t_enc_logits, t_pred_future_logits, t_pred_past_logits):
        # update batch centers
        batch_center = self.get_batch_center(t_enc_logits)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        batch_center_pred_future = self.get_batch_center(t_pred_future_logits)
        self.predict_future_center = self.predict_future_center * self.center_momentum \
                                     + batch_center_pred_future * (1 - self.center_momentum)

        batch_center_pred_past = self.get_batch_center(t_pred_past_logits)
        self.predict_past_center = self.predict_past_center * self.center_momentum \
                                   + batch_center_pred_past * (1 - self.center_momentum)

        return {'entropy': self.entropy(self.center),
                'future_entropy': self.entropy(self.predict_future_center),
                'past_entropy': self.entropy(self.predict_past_center)}

    @torch.no_grad()
    def get_batch_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        b, t, *_ = teacher_output.shape
        batch_center = torch.sum(torch.sum(teacher_output, dim=0, keepdim=True), dim=1)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (b * t * dist.get_world_size())
        return batch_center
