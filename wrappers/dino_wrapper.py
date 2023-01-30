__all__ = ['MultiCropWrapper', 'MultiCropWrapperDino']

import torch
from torch import nn


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head, predictor, predictor_past=None, headprob=None, **kwargs):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        if hasattr(backbone, 'fc'):
            backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.predictor = predictor
        self.predictor_past = predictor_past
        self.headprob = headprob

    def forward(self, x, **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
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
        # Run the head forward on the concatenated features.
        if self.training:
            return self.head(output)
        else:
            return output


from .base_wrapper import MultiCropWrapperBase


class MultiCropWrapperDino(MultiCropWrapperBase):
    def forward_student(self, x_enc, m_enc=None, m_mask=None, **kwargs):
        s_enc_logits = self.forward_encode(x_enc[:, -1:])
        return s_enc_logits,

    def forward_teacher(self, x_enc, video_indices=None, **kwargs):
        t_enc_logits = self.forward_encode(x_enc[:, :-1])
        return t_enc_logits,
