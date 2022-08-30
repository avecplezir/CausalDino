__all__ = ['NextTokenLoss']

import torch
from .feature_loss import FeatureLoss


class NextTokenLoss(FeatureLoss):
    def compute_loss_fe(self, s_pred_future_proba, t_enc_proba):
        CE_fe = torch.sum(-t_enc_proba[:, 1:] * torch.log(s_pred_future_proba[:, :-1]), dim=-1).mean()
        return CE_fe

    def compute_loss_ef(self, s_enc_proba, t_pred_future_proba):
        CE_ef = torch.sum(-t_pred_future_proba[:, :-1] * torch.log(s_enc_proba)[:, 1:], dim=-1).mean()
        return CE_ef
