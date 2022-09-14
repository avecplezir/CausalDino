__all__ = ['BertLoss']

import torch
import torch.nn.functional as F
import numpy as np

from .feature_loss import FeatureLoss


class BertLoss(FeatureLoss):
    def forward(self, student_output, teacher_output, epoch, student=None, teacher=None, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc, _ = student_output
        t_enc, t_indices = teacher_output

        temp = self.teacher_temp_schedule[epoch]
        if self.args.teacher_prediction_type == 'head_predictor_joint':
            t_enc_logits = teacher.head(teacher.predictor(s_enc, attn_type='id'))
        elif self.args.teacher_prediction_type == 'predictor':
            t_enc_logits = teacher.headproba(teacher.predictor(s_enc, attn_type='id'))
        elif self.args.teacher_prediction_type == 'head':
            t_enc_logits = teacher.head(s_enc)
        else:
            assert 0, f'{self.args.teacher_prediction_type} not implemented!'

        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)

        CE_fe = self.compute_loss_fe(s_enc, t_enc_proba, student, t_indices) if self.args.CE_fe_c else 0
        total_loss = CE_fe

        self.update_centers(t_enc_logits, None, None)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    def generate_masks(self, pos_indices):
        b, T = pos_indices.size()
        binT = lambda x: ''.join(reversed([str((x >> i) & 1) for i in range(T)]))
        masks = []
        for idx in range(1, 2 ** T - 1):
            masks.append(binT(idx))
        masks = np.array(masks, dtype=int)
        return torch.tensor(masks).to(pos_indices.device)

    def compute_loss_fe(self, s_enc, t_enc_proba, student, pos_indices):
        total_loss = 0
        n_loss_terms = 0
        masks = self.generate_masks(pos_indices)
        print('masks', masks)
        print('s_enc', s_enc.shape)
        for mask in masks:
            print('mask', mask)
            mask = mask.unsqueeze(0)
            if self.args.student_prediction_type == 'predictor_first':
                s_pred_future = student.module.predictor(s_enc, indices=pos_indices, mask=mask,
                                                         attn_type='all')
                s_pred_future_logits = student.module.head(s_pred_future)
            elif self.args.student_prediction_type == 'head_first':
                s_enc = student.module.head(s_enc)
                s_pred_future_logits = student.module.predictor(s_enc,
                                                                indices=pos_indices, mask=mask,
                                                                attn_type='all')
            else:
                assert 0, f'{self.args.student_prediction_type} not implemented!'

            print('s_pred_future_proba', s_pred_future_proba.shape)
            print('t_enc_proba', t_enc_proba.shape)

            s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)
            loss = -torch.sum(t_enc_proba * torch.log(s_pred_future_proba), dim=-1)
            inverse_mask = (~mask.bool()).long()
            total_loss += (inverse_mask * loss).sum()
            n_loss_terms += inverse_mask.sum()

        total_loss /= n_loss_terms
        return total_loss
