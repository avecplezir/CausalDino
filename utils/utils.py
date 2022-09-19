# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque
import io
import wandb

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps

try:
    import yt.wrapper as yt
    from yt.wrapper import file_commands
except:
    pass


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def synchronize():
    if not is_dist_avail_and_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def restart_from_pretrain(ckp_path, **kwargs):
    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    for key, value in kwargs.items():
        try:
            msg = value.load_state_dict(checkpoint, strict=False)
            print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
        except TypeError:
            try:
                msg = value.load_state_dict(checkpoint)
                print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
            except ValueError:
                print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return wandb.util.generate_id()
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint.get(var_name)

    wandb_id = (checkpoint["args"].wandb_id if ("args" in checkpoint and "wandb_id" in checkpoint["args"])
                else wandb.util.generate_id())
    return wandb_id


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def load_checkpoint_from_yt(path) -> dict:
    print(f"Loading checkpoint from {path}")
    yt_client = yt.YtClient('hahn')
    stream = yt_client.read_file(path)
    buf = io.BytesIO(stream.read())
    checkpoint = torch.load(buf, map_location="cpu")
    return checkpoint


def make_client(proxy='hahn', max_thread_count=20, extra_config=None):
    config = {
        'read_retries': {'enable': True},
        'proxy': {'accept_encoding': 'gzip', 'content_encoding': 'gzip'}
    }

    if max_thread_count > 1:
        config['read_parallel'] = {
            'enable': True,
            'max_thread_count': max_thread_count,
        }

    if extra_config is not None:
        config.update(extra_config)

    return yt.YtClient(proxy=proxy, config=config)


# def transfer_checkpoint_to_yt(path='/home/ivananokhin/.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth',
#                               yt_path='//home/yr/ianokhin/CausalDino/pretrain/dino_deitsmall16_pretrain.pth'):
# def transfer_checkpoint_to_yt(path='/home/ivananokhin/.cache/torch/hub/checkpoints/mae_pretrain_vit_base.pth',
#                                   yt_path='//home/yr/ianokhin/CausalDino/pretrain/mae_pretrain_vit_base.pth'):
def transfer_checkpoint_to_yt(path='/home/ivananokhin/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth',
                                  yt_path='//home/yr/ianokhin/CausalDino/pretrain/dino_vitbase16_pretrain.pth'):
    state_dict = torch.load(path)
    print("Saving checkpoint")
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    data = buf.getvalue()

    yt_client = make_client()
    file_commands.write_file(yt_path, data, client=yt_client)


def save_checkpoint_to_yt(args, state_dict, epoch=None):
    "Save checkpoint (optionaly write to yt table)"
    if is_main_process() and args.yt_path is not None:
        experiment_path = os.path.join(args.yt_path, 'CausalDino', args.exp_name)
        checkpoint_path = os.path.join(experiment_path, "checkpoint.pt")
        experiment_path, _ = os.path.split(checkpoint_path)

        yt_client = make_client()
        print("Saving checkpoint")
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        data = buf.getvalue()
        if epoch is not None:
            checkpoint_path = os.path.join(experiment_path, f"checkpoint_epoch{epoch}.pt")
        file_commands.write_file(checkpoint_path, data, client=yt_client)

    if is_main_process():
        try:
            import nirvana_dl.snapshot as snap
            snap.dump_snapshot()
            print('Checkpoint saved to snapshots.')
        except Exception:
            print('Checkpoint NOT save to snapshots!')
            pass


def restore_yt_checkpoint(args):
    if is_main_process() and args.yt_path is not None:
        experiment_path = os.path.join(args.yt_path, 'CausalDino', args.exp_name)
        checkpoint_path = os.path.join(experiment_path, "checkpoint.pt")
        yt_client = yt.YtClient("hahn")
        if not yt_client.exists(f"{experiment_path}/tmp"):
            yt_client.mkdir(f"{experiment_path}/tmp", recursive=True)

        if yt_client.exists(checkpoint_path):
            checkpoint = load_checkpoint_from_yt(checkpoint_path)
            torch.save(checkpoint, "checkpoint.pt")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


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


class Memory:
    def __init__(self, batch_size, maxlen):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.memory = deque(maxlen=self.maxlen)
        self.memory_mask = deque(maxlen=self.maxlen)
        self.current_video_indices = -torch.ones(batch_size)
        self.memory_idx = 0

    def add(self, values):
        values = values[:, self.memory_idx]
        self.memory.append(values.detach())
        self.memory_mask.append(torch.ones(self.batch_size).to(values.device))

    def remove(self, video_indices):
        video_indices = video_indices.cpu()
        new_video_indices = ~(self.current_video_indices == video_indices)
        self.current_video_indices = video_indices
        for idx in torch.arange(self.batch_size)[new_video_indices]:
            for i in range(len(self.memory)):
                self.memory[i][idx] = torch.zeros_like(self.memory[i][idx])
                self.memory_mask[i][idx] = 0

    def retrieve(self, ):
        return torch.stack(list(self.memory), 1), torch.stack(list(self.memory_mask), 1)


class PatchMemory:
    def __init__(self, batch_size, maxlen):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.memory = deque(maxlen=self.maxlen)
        self.memory_mask = deque(maxlen=self.maxlen)
        self.current_video_indices = -torch.ones(batch_size)

    def add(self, values):
        self.memory.append(values.detach())
        self.memory_mask.append(torch.ones(self.batch_size, values.size(1)).to(values.device))

    def remove(self, video_indices):
        video_indices = video_indices.cpu()
        new_video_indices = ~(self.current_video_indices == video_indices)
        self.current_video_indices = video_indices
        for idx in torch.arange(self.batch_size)[new_video_indices]:
            for i in range(len(self.memory)):
                self.memory[i][idx] = torch.zeros_like(self.memory[i][idx])
                self.memory_mask[i][idx] = 0

    def retrieve(self, ):
        return torch.cat(list(self.memory), 1), torch.cat(list(self.memory_mask), 1)


class MultiCropWrapperGeneral(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head, predictor, predictor_past=None,
                 headprob=None, args=None, mode=None, loss_mode=None, memory=None, **kwargs):
        super(MultiCropWrapperGeneral, self).__init__()
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

    def forward_teacher(self, x_enc, headprob=True):
        if self.args.teacher_prediction_type == 'head_predictor_joint':
            indices = torch.zeros_like(x_enc[:, :, 0]).long()
            t_enc = self.head(self.predictor(x_enc, indices=indices, attn_type='id'))
        elif self.args.teacher_prediction_type == 'head':
            t_enc = self.head(x_enc)
        else:
            assert 0, f'{self.args.teacher_prediction_type} not implemented!'
        if headprob:
            t_enc_logits = self.headprob(t_enc)
            return t_enc_logits
        else:
            return t_enc

    def forward_student_gpt(self, x_enc, indices, mask=None):
        if self.args.student_prediction_type == 'predictor_first':
            s_pred = self.predictor(x_enc, indices=indices, mask=mask)
            s_pred_logits = self.headprob(self.head(s_pred))
        elif self.args.student_prediction_type == 'head_first':
            s_enc_head = self.head(x_enc)
            s_pred = self.predictor(s_enc_head, indices=indices, mask=mask)
            s_pred_logits = self.headprob(s_pred)
        else:
            assert 0, f'{self.args.student_prediction_type} not implemented!'
        return s_pred_logits

    def forward_student_vae(self, x_enc, indices):
        if self.args.student_prediction_type == 'predictor_first':
            s_pred, stoch_post, stats_post, stats_prior = self.predictor(x_enc, indices=indices)
            s_pred_logits = self.headprob(self.head(s_pred))
        elif self.args.student_prediction_type == 'head_first':
            s_enc_head = self.head(x_enc)
            s_pred, stoch_post, stats_post, stats_prior = self.predictor(s_enc_head, indices=indices)
            s_pred_logits = self.headprob(s_pred)
        else:
            assert 0, f'{self.args.student_prediction_type} not implemented!'
        return s_pred_logits, stoch_post, stats_post, stats_prior

    def forward_student_mask(self, x_enc, indices, mask):
        if self.args.student_prediction_type == 'predictor_first':
            s_pred = self.predictor(x_enc, indices=indices, mask=mask, attn_type='all')
            s_pred_logits = self.headprob(self.head(s_pred))
        elif self.args.student_prediction_type == 'head_first':
            s_enc_head = self.head(x_enc)
            s_pred = self.predictor(s_enc_head, indices=indices, mask=mask, attn_type='all')
            s_pred_logits = self.headprob(s_pred)
        else:
            assert 0, f'{self.args.student_prediction_type} not implemented!'
        return s_pred_logits

    def forward_student_bert(self, x_enc, indices):
        s_pred_logits_list = []
        masks = self.generate_masks(indices)
        for mask in masks:
            mask = mask.unsqueeze(0)
            s_pred_logits = self.forward_student_mask(x_enc, indices, mask)
            s_pred_logits_list.append(s_pred_logits)
        return s_pred_logits_list, masks

    def forward_timeemb(self, x_enc, indices):
        s_pred_logits_list = []
        x_enc_head = self.head(x_enc)
        for ie in range(1, self.args.n_global_views):  # future encoding
            s_pred = self.predictor(x_enc_head[:, :ie], future_index=indices[:, ie],
                                    indices=indices[:, :ie])[:, 1:]
            s_pred_logits = self.headprob(s_pred)
            s_pred_logits_list.append(s_pred_logits)
        return s_pred_logits_list

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
                if self.loss_mode == 'memory_bert':
                    t_enc_head = self.forward_teacher(x_enc, headprob=False)
                    self.memory.add(t_enc_head)
                    self.memory.remove(video_indices)
                    memory_enc, memory_mask = self.memory.retrieve()
                    t_m_enc_logits = self.headprob(memory_enc)
                    t_enc_logits = self.headprob(t_enc_head)
                    return t_m_enc_logits, t_enc_logits, memory_mask, memory_enc
                elif self.loss_mode == 'memory_vae':
                    self.memory.add(x_enc)
                    self.memory.remove(video_indices)
                    memory_enc, memory_mask = self.memory.retrieve()
                    t_m_enc_logits = self.forward_teacher(memory_enc)
                    t_enc_logits = self.forward_teacher(x_enc)
                    return t_m_enc_logits, t_enc_logits, memory_mask, memory_enc
                elif self.loss_mode == 'memory_gpt':
                    self.memory.add(x_enc)
                    self.memory.remove(video_indices)
                    memory_enc, memory_mask = self.memory.retrieve()
                    t_enc_logits = self.forward_teacher(x_enc)
                    return t_enc_logits, memory_mask, memory_enc
                else:
                    return self.forward_teacher(x_enc)
            elif self.mode == 'student':
                if self.loss_mode == 'gpt':
                    return self.forward_student_gpt(x_enc, indices), None
                if self.loss_mode == 'bert':
                    return self.forward_student_bert(x_enc, indices)
                elif self.loss_mode == 'vae':
                    s_pred_logits, stoch_post, stats_post, stats_prior = self.forward_student_vae(x_enc, indices)
                    return s_pred_logits, stats_post, stats_prior
                elif self.loss_mode == 'timeemb':
                    return self.forward_timeemb(x_enc, indices), None
                elif self.loss_mode == 'memory_bert':
                    bert_mask, bert_indices = self.get_memory_bert_indices_mask(indices)
                    bert_x_enc = torch.cat([torch.zeros_like(x_enc[:, :1].repeat(1, self.args.maxlen-1, 1)), x_enc[:, :1]], 1)
                    s_m_pred_logits = self.forward_student_mask(bert_x_enc, bert_indices, bert_mask)
                    s_pred_logits = self.forward_teacher(x_enc) if self.args.CE_ee_c else 0.
                    return s_m_pred_logits, bert_mask, s_pred_logits
                elif self.loss_mode == 'memory_vae':
                    x_enc = torch.cat([m_enc[:, :-1], x_enc[:, :1]], 1)
                    indices = self.get_indices(x_enc, maxlen=False)
                    s_pred_logits, stoch_post, stats_post, stats_prior = self.forward_student_vae(x_enc, indices)
                    return s_pred_logits, stats_post, stats_prior
                elif self.loss_mode == 'memory_gpt':
                    t = x_enc.size(1)
                    x_enc = torch.cat([m_enc[:, :-t], x_enc], 1)
                    indices = self.get_indices(x_enc, maxlen=False)
                    s_pred_logits = self.forward_student_gpt(x_enc, indices, mask=m_mask)
                    return s_pred_logits[:, -t:], None
                else:
                    assert 0, f'mode {self.loss_mode} not implemented'
            else:
                assert 0, f'mode {self.mode} not implemented!'
        else:
            return output

    def get_indices(self, x, maxlen=True):
        t = self.args.maxlen if maxlen else x.size(1)
        return torch.arange(t).flip([0]).unsqueeze(0).to(x.device)

    def get_memory_bert_indices_mask(self, x):
        bert_indices = self.get_indices(x)
        bert_mask = torch.zeros_like(bert_indices).repeat(x.size(0), 1)
        bert_mask[:, -1] = 1
        return bert_mask, bert_indices

    def generate_masks(self, pos_indices):
        b, T = pos_indices.size()
        binT = lambda x: ''.join(reversed([str((x >> i) & 1) for i in range(T)]))
        masks = []
        for idx in range(1, 2 ** T - 1):
            mask = np.array(list(binT(idx)), dtype=int)
            # if 1 < sum(mask) < 3:
            masks.append(mask)
        masks = np.array(masks, dtype=int)
        # masks = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]
        return torch.tensor(masks).to(pos_indices.device)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def get_diff_images(images, idx=None):
    if idx is None:
        return [im[:, :, 1:, ...] - im[:, :, :-1, ...] for im in images]

    else:
        return [im[:, :, idx + 1, :, :] - im[:, :, idx, :, :] for im in images]


def get_flow_images(images, temporal_length=8):
    out_list = []
    for im in images:
        idx = np.random.randint(0, temporal_length)
        out_list.append(im[:, :, idx, ...])
    return out_list
