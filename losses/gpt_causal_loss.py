__all__ = ['GPTCausalLoss']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class GPTCausalLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, two_token=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.two_token = two_token
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("prediction_center", torch.zeros(1, out_dim))
        self.register_buffer("prediction_past_center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    # predict_proba predict_future predict_past

    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out_list = student_output.chunk(self.n_crops)
        student_out = torch.stack(student_out_list, 1)

        teacher_out_list = teacher_output.detach().chunk(self.n_crops)
        teacher_out = torch.stack(teacher_out_list, 1)
        temp = self.teacher_temp_schedule[epoch]

        # Student
        # Encoding
        student_enc_proba = F.softmax(student.predict_logits(student_out) / self.student_temp, dim=-1)
        # Prediction
        student_pred_logits = student.predict_logits(student.predict_future(student_out))
        student_pred_proba = F.softmax(student_pred_logits / self.student_temp, dim=-1)
        # Inverse
        student_out_inv = torch.stack(student_out_list[::-1], 1)
        student_pred_inv_logits = student.predictor_past(student_out_inv)
        student_pred_inv_proba = F.softmax(student_pred_inv_logits / self.student_temp, dim=-1)

        # Teacher
        # Encoding
        teacher_enc_logits = teacher.predict_logits(teacher_out)
        teacher_enc_proba = F.softmax((teacher_enc_logits - self.center) / temp, dim=-1)
        # Prediction
        teacher_pred_logits = teacher.predict_logits(teacher.predictor(teacher_out))
        teacher_pred_proba = F.softmax((teacher_pred_logits - self.prediction_center) / temp, dim=-1)
        # Inverse
        teacher_out_inv = torch.stack(teacher_out_list[::-1], 1)
        teacher_pred_inv_logits = teacher.predict_logits(teacher.predictor_past(teacher_out_inv))
        teacher_pred_inv_proba = F.softmax((teacher_pred_inv_logits - self.prediction_past_center) / temp, dim=-1)

        # Losses
        # allign future student prediction with teacher encoding (weighted with past teacher prediction)
        CE_fp = self.compute_loss(student_pred_proba, teacher_enc_proba, teacher_pred_inv_proba)
        # allign future teacher prediction with student encoding (weighted with past teacher prediction)
        CE_pf = self.compute_loss(student_enc_proba, teacher_pred_proba, teacher_pred_inv_proba)
        # allign past student prediction with teacher encoding (weighted with future teacher prediction)
        CE_fp_inv = self.compute_loss(student_pred_inv_proba, teacher_enc_proba, teacher_pred_proba)
        # allign past teacher prediction with student encoding (weighted with future teacher prediction)
        CE_pf_inv = self.compute_loss(student_enc_proba, teacher_pred_inv_proba, teacher_pred_proba)

        CE_fp = CE_fp.mean()
        CE_pf = CE_pf.mean()
        CE_fp_inv = CE_fp_inv.mean()
        CE_pf_inv = CE_pf_inv.mean()
        total_loss = 0.45*CE_fp + 0.05*CE_pf + 0.45*CE_fp_inv + 0.05*CE_pf_inv

        # update batch centers
        batch_center = self.get_batch_center(teacher_enc_logits)
        self.update_center(batch_center)
        true_entropy = torch.sum(F.softmax(self.center, dim=-1) * F.log_softmax(self.center), dim=-1)
        entropy = torch.sum(F.softmax(batch_center, dim=-1) * F.log_softmax(self.center), dim=-1)

        batch_center_pred = self.get_batch_center(teacher_pred_logits)
        self.update_prediction_center(batch_center_pred)
        true_entropy_prediction = torch.sum(F.softmax(self.center, dim=-1) * F.log_softmax(self.center), dim=-1)
        entropy_prediction = torch.sum(F.softmax(batch_center_pred, dim=-1) * F.log_softmax(self.center), dim=-1)

        batch_center_pred_past = self.get_batch_center(teacher_pred_inv_logits)
        self.update_prediction_past_center(batch_center_pred_past)

        return total_loss, {'CE': total_loss, 'CE_fp': CE_fp, 'CE_pf': CE_pf,
                            'CE_fp_inv': CE_fp_inv, 'CE_pf_inv': CE_pf_inv,
                            'entropy': entropy, 'true_entropy': true_entropy,
                            'true_entropy_prediction': true_entropy_prediction,
                            'entropy_prediction': entropy_prediction}

    def compute_loss(self, prediction, labels, inverse):
        total_loss = 0
        n_loss_terms = 0
        for ip, p in enumerate(prediction.chunk(self.n_crops, dim=1)): # past
            for il in range(ip + 1, len(labels)): #future
                loss = -torch.sum(torch.sum(labels[:, il] * torch.log(p/(1 - inverse[:, ip] + 1e-4)), dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

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

    @torch.no_grad()
    def update_center(self, batch_center):
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return batch_center

    @torch.no_grad()
    def update_prediction_center(self, batch_center):
        """
        Update center used for teacher output.
        """
        # ema update
        self.prediction_center = self.prediction_center * self.center_momentum + batch_center * (1 - self.center_momentum)


    @torch.no_grad()
    def update_prediction_past_center(self, batch_center):
        """
        Update center used for teacher output.
        """
        # ema update
        self.prediction_past_center = self.prediction_past_center * self.center_momentum + batch_center * (1 - self.center_momentum)
