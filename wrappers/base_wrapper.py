__all__ = ['MultiCropWrapperBase']

import torch
from torch import nn


class MultiCropWrapperBase(nn.Module):
    def __init__(self, backbone, head, predictor, predictor_past=None,
                 headprob=None, args=None, mode=None, loss_mode=None, memory=None, **kwargs):
        super(MultiCropWrapperBase, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        if hasattr(backbone, 'fc'):
            backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.predictor = predictor
        self.predictor_past = predictor_past
        self.headprob = headprob
        self.memory = memory
        self.args = args
        self.mode = mode
        self.loss_mode = loss_mode

    def forward_backbone(self, x, **kwargs):
        # convert to list
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]), **kwargs)
            if start_idx == 0:
                output = _out
            else:
                if isinstance(_out, tuple):
                    output1 = torch.cat((output[0], _out[0]))
                    output2 = torch.cat((output[1], _out[1]))
                    output = (output1, output2)
                else:
                    output = torch.cat((output, _out))
            start_idx = end_idx
        return output

    def forward_encode(self, x_enc, headprob=True):
        if self.args.teacher_prediction_type == 'head_predictor_joint':
            indices = torch.zeros_like(x_enc[:, :, 0]).long()
            t_pred = self.predictor(x_enc, indices=indices, attn_type='id')
            t_enc = self.head(t_pred)
        elif self.args.teacher_prediction_type == 'head':
            t_enc = self.head(x_enc)
        else:
            assert 0, f'{self.args.teacher_prediction_type} not implemented!'
        if headprob:
            t_enc_logits = self.headprob(t_enc)
            return t_enc_logits
        else:
            return t_enc

    def forward_predict(self, x_enc, indices=None, mask=None, attn_type='causal', future_index=None):
        if self.args.student_prediction_type == 'predictor_first':
            s_pred = self.predictor(x_enc, indices=indices, mask=mask, attn_type=attn_type,
                                    future_index=future_index)
            s_pred_logits = self.headprob(self.head(s_pred))
        elif self.args.student_prediction_type == 'head_first':
            s_enc_head = self.head(x_enc)
            s_pred = self.predictor(s_enc_head, indices=indices, mask=mask, attn_type=attn_type,
                                    future_index=future_index)
            s_pred_logits = self.headprob(s_pred)
        else:
            assert 0, f'{self.args.student_prediction_type} not implemented!'
        return s_pred_logits

    def forward_student(self, x_enc, indices, **kwargs):
        s_enc_logits = self.forward_encode(x_enc)
        s_pred_logits = self.forward_predict(x_enc, indices)
        return s_pred_logits, s_enc_logits

    def forward_teacher(self, x_enc, indices, **kwargs):
        t_enc_logits = self.forward_encode(x_enc)
        t_pred_logits = self.forward_predict(x_enc, indices)
        return t_pred_logits, t_enc_logits

    def forward(self, x, indices=None, video_indices=None, m_enc=None, m_mask=None, **kwargs):
        if not isinstance(x, list):
            x = [x]
        n_crops = len(x)
        output = self.forward_backbone(x, **kwargs)
        # Run the head forward on the concatenated features.
        if self.training:
            enc_list = output.chunk(n_crops)
            x_enc = torch.stack(enc_list, 1)
            if self.mode == 'teacher':
                return self.forward_teacher(x_enc=x_enc, indices=indices, video_indices=video_indices)
            elif self.mode == 'student':
                return self.forward_student(x_enc=x_enc, indices=indices, video_indices=video_indices,
                                            m_enc=m_enc, m_mask=m_mask)
            else:
                assert 0, f'mode {self.mode} not implemented!'
        else:
            return output
    