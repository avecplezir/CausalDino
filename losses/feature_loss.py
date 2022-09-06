__all__ = ['FeatureLoss']

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from .dino_loss import DINOLoss


class FeatureLoss(DINOLoss):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, args=None, n_global_views=None,
                 start_video_idx=None, video_clip_size=None, index2clip_video=None,
                 **kwargs):
        super(DINOLoss, self).__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.args = args
        self.n_global_views = n_global_views
        self.register_buffer("center", torch.zeros(1, 1, out_dim))
        self.register_buffer("predict_future_center", torch.zeros(1, 1, out_dim))
        self.register_buffer("predict_past_center", torch.zeros(1, 1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        if self.args.continuous:
            self.start_video_idx = start_video_idx
            self.video_clip_size = video_clip_size
            self.index2clip_video = index2clip_video
            self.memory = None
            self.init_memory(video_clip_size)

    def init_memory(self, video_clip_size):
        # self.register_buffer("memory", -torch.ones(sum(video_clip_size)))
        self.memory = -np.ones(sum(video_clip_size))

    def add_memory(self, keys, values):
        if self.args.continuous:
            self.memory[keys] = values

    def retrieve_memory(self, keys):
        if self.args.continuous:
            return self.memory[keys]

    def forward(self, student_output, teacher_output, epoch, **kwargs):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        s_enc_logits, s_pred_future_logits, s_pred_past_logits, s_indices = student_output
        t_enc_logits, t_pred_future_logits, t_pred_past_logits, t_indices = teacher_output

        # retrieved_memories = self.retrieve_memory(t_indices.cpu().numpy())

        temp = self.teacher_temp_schedule[epoch]

        s_enc_proba = F.softmax(s_enc_logits / self.student_temp, dim=-1)
        s_pred_future_proba = F.softmax(s_pred_future_logits / self.student_temp, dim=-1)

        t_enc_proba = F.softmax((t_enc_logits - self.center) / temp, dim=-1)
        t_pred_future_proba = F.softmax((t_pred_future_logits - self.predict_future_center) / temp, dim=-1)

        CE_fe = self.compute_loss_fe(s_pred_future_proba, t_enc_proba)
        CE_ef = self.compute_loss_ef(s_enc_proba, t_pred_future_proba)

        total_loss = self.args.CE_fe_c * CE_fe + self.args.CE_ef_c * CE_ef

        # self.add_memory(t_indices.view(-1).cpu().numpy(), t_enc_logits.argmax(dim=-1).view(-1).cpu().numpy())

        self.update_centers(t_enc_logits, t_pred_future_logits, t_pred_past_logits)
        time_entropy = self.time_entropy(t_enc_proba)
        dirac_entropy, dirac_entropy_proportion2max = self.dirac_entropy(t_enc_logits)

        return total_loss, {'CE': total_loss,
                            'CE_fe': CE_fe,
                            'CE_ef': CE_ef,
                            'entropy': self.entropy(self.center),
                            'batch_time_entropy': time_entropy,
                            'dirac_entropy': dirac_entropy,
                            'dirac_entropy_proportion2max': dirac_entropy_proportion2max,
                            }

    @torch.no_grad()
    def update_centers(self, t_enc_logits, t_pred_future_logits, t_pred_past_logits):
        # update batch centers
        batch_center = self.get_batch_center(t_enc_logits)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        if t_pred_future_logits is not None:
            batch_center_pred_future = self.get_batch_center(t_pred_future_logits)
            self.predict_future_center = self.predict_future_center * self.center_momentum \
                                         + batch_center_pred_future * (1 - self.center_momentum)

        if t_pred_past_logits is not None:
            batch_center_pred_past = self.get_batch_center(t_pred_past_logits)
            self.predict_past_center = self.predict_past_center * self.center_momentum \
                                       + batch_center_pred_past * (1 - self.center_momentum)

    def compute_kl(self, conditional):
        marginal_log = F.log_softmax(self.center.detach(), dim=-1).repeat(conditional.size(0), conditional.size(1), 1)
        conditional_log = torch.log(conditional)
        kl = F.kl_div(conditional_log, marginal_log, log_target=True)
        return kl.mean()

    def compute_loss_fe(self, s_pred_future_proba, t_enc_proba):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ip in range(0, self.n_global_views): #future_prediction from past
            for ie in range(ip + 1, self.n_global_views): #future encoding
                loss = -torch.sum(t_enc_proba[:, ie] * torch.log(s_pred_future_proba[:, ip]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    def compute_loss_ef(self, s_enc_proba, t_pred_future_proba):
        total_loss = 0
        n_loss_terms = 0
        # ip < ie
        for ip in range(0, self.n_global_views): #future_prediction from past
            for ie in range(ip + 1, self.n_global_views): #future encoding
                loss = -torch.sum(t_pred_future_proba[:, ip] * torch.log(s_enc_proba[:, ie]), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    def get_batch_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        b, t, *_ = teacher_output.shape
        batch_center = torch.sum(torch.sum(teacher_output, dim=0, keepdim=True), dim=1,  keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (b * t * dist.get_world_size())
        return batch_center

