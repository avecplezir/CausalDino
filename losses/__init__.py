from .dino_loss import DINOLoss
from .feature_loss import FeatureLoss, FeatureLossAllPairs, ByolLossAllPairs
from .next_token_loss import NextTokenLoss
from .timeemb_loss import TimeEmbLoss
from .dino_gumbel_loss import DINOGumbelLoss, DINOGumbel2Loss, DINOTopkLoss, DINOGumbel3Loss, DINORandomChoiceLoss
from .te_pp_loss import TEPPLoss
from .vae_loss import VAELoss
from .memory_loss import MemoryLoss
from .memory_bert_loss import MemoryBertLoss
from .memory_past_loss import MemoryPastLoss, MemoryVAELoss
from .tepp_loc_loss import TEPPLocLoss
from .bert_loss import BertLoss

