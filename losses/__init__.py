# usual dino
from .one_dir.dino_loss import DINOLoss
# gradinet flows only in predictor branch
from .one_dir.feature_loss import FeatureLoss, FeatureLossAllPairs, ByolLossAllPairs
from .one_dir.base_losses import BertLoss, GPTLoss, TELoss
from .one_dir.memory_losses import GPTMemoryLoss, MemoryLoss
# gradinet flows in both directions (predictor branch and encoding branch)
from .two_dir.base_losses import GPT2Loss, TE2Loss
from .two_dir.memory_losses import GPT2MemoryLoss, TE2MemoryLoss
