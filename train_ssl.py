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
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import wandb
import logging

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models

from utils import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from models import SwinTransformer3D


from utils.parser import load_config
from eval_knn import extract_features, knn_classifier
import losses
import datasets
import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('SVT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small', 'timesformer',
                                 'swin'] + torchvision_archs,
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--pretrained_rgb', default=None, type=str, help='Path to pretrained RGB model.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    # online knn eval
    parser.add_argument('--eval_batch_size_per_gpu', default=32, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=5, type=int, help='Number of NN to use. We use 5 for online.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')

    parser.add_argument('--exp_name', default='svt', type=str, help='Experiment name.')
    parser.add_argument("--log_every", type=int, default=20, help="Log loss every")
    parser.add_argument('--do_eval', type=utils.bool_flag, default=False, help="""Whether to do knn eval.""")
    parser.add_argument('--do_eval_before_train', type=utils.bool_flag, default=False, help="""Whether to do knn eval before train.""")
    parser.add_argument('--loss', default=None, type=str, help="""Name of loss to train with.""")
    parser.add_argument('--dataset', default=None, type=str, help="""Name of dataset to train with.""")
    parser.add_argument('--use_wandb', type=utils.bool_flag, default=True, help="""Whether to log with wandb.""")
    parser.add_argument('--skip_last', type=utils.bool_flag, default=False,
                        help="""Whether to skip last layer in dino head.""")
    parser.add_argument("--n_parts", type=int, default=9, help="For how many parts initial video is divided")
    parser.add_argument("--n_global_views", type=int, default=2, help="Number of global views to sample")
    parser.add_argument('--random_sampling', type=utils.bool_flag, default=True, help="""Whether random sampling video chunks.""")
    parser.add_argument('--predictor', default=None, type=str, help="""Name of predictor to train with.""")
    parser.add_argument('--predictor_past', default=None, type=str, help="""Name of predictor Inverse to train with.""")
    parser.add_argument('--headproba', default=None, type=str, help="""Name of probability head to train with.""")
    parser.add_argument('--wrapper', default='MultiCropWrapper', type=str, help="""Name of wrapper to train with.""")
    parser.add_argument('--argmax', type=utils.bool_flag, default=False,
                        help="""Whether to find inv with argmax.""")
    parser.add_argument('--weight_inv', type=utils.bool_flag, default=True,
                        help="""Whether to use inv in loss.""")
    parser.add_argument('--eval_dataset', default='UCFReturnIndexDataset', type=str, help="""Name of dataset to test knn with.""")
    parser.add_argument('--model_name', default=None, type=str, help="""Name of model""")
    parser.add_argument("--eval_freq", type=int, default=2, help="eval every")
    parser.add_argument('--eval_dataset2', default='UCFEventsReturnIndexDataset', type=str,
                        help="""Name of dataset to test knn with.""")
    parser.add_argument('--video_extension', default='avi', type=str, help='Video extension.')
    parser.add_argument('--CE_fe_c', default=0.5, type=float, help='loss coefficient')
    parser.add_argument('--CE_ef_c', default=0.5, type=float, help='loss coefficient')
    parser.add_argument('--kl_c', default=1., type=float, help='loss coefficient')
    parser.add_argument('--CE_ee_c', default=0.5, type=float, help='loss coefficient')
    parser.add_argument('--coef_entropy', default=1, type=float, help='loss coefficient')
    parser.add_argument("--bottleneck_dim", type=int, default=256, help="bottleneck dim in Dino Head")
    parser.add_argument('--predictor_model_type', default='gpt-micro-256', type=str, help="""Name of model""")
    parser.add_argument('--yt_path', default=None, type=str, help="Base path for logs storage in yt")
    parser.add_argument('--val_data_dir', default="/mnt/data/UCF101", type=str,
                        help="path to validation dataset")  # "//home/yr/ianokhin"
    parser.add_argument('--dataset_level', default=1, type=int,
                        help="level to read the data")
    parser.add_argument('--continuous', type=utils.bool_flag, default=False,
                        help="""Whether to use continuous sampler""")
    parser.add_argument('--return_prediction_logits', type=utils.bool_flag, default=True,
                        help="""Whether to return logits with prediction""")
    parser.add_argument('--return_enc_logits', type=utils.bool_flag, default=True,
                        help="""Whether to return logits of encoding""")
    parser.add_argument('--pseudo_length', type=int, default=None,
                        help="""pseudo_length of the dataset""")
    parser.add_argument('--sampling_rate', type=int, default=32,
                        help="""sampling rate for video reading""")
    parser.add_argument('--global_size', type=int, default=224,
                        help="""size of the image with global augmentation""")
    parser.add_argument('--num_frames', type=int, default=8,
                        help="""number of frames in each clip""")
    parser.add_argument('--default_cfg', default=None, type=str, help='Video extension.')
    parser.add_argument('--full_pretrain', default=None, type=str, help='path to pretrained checkpoint')
    parser.add_argument('--block_size', type=int, default=64,
                        help="""block_size in gpt""")
    parser.add_argument('--maxlen', type=int, default=8,
                        help="""max len in memory""")
    parser.add_argument('--kl_balance', default=0.8, type=float, help='kl balance in VAE loss')
    parser.add_argument('--temporal_aug_memory', type=utils.bool_flag, default=False,
                        help="""Whether to use continuous sampler""")
    parser.add_argument('--memory_balance_loss', type=utils.bool_flag, default=False,
                        help="""Whether to use memory balancing""")
    parser.add_argument('--teacher_pred_head', type=utils.bool_flag, default=False,
                        help="""Whether to use teacher prediction head in memory loss""")
    parser.add_argument('--teacher_enc_pred_head', type=utils.bool_flag, default=False,
                        help="""Whether to use teacher prediction head in memory loss in encoder""")
    parser.add_argument('--masking_ratio', default=0.2, type=float, help='ratio of masked tokens for bert-like loss')
    parser.add_argument('--memory_offset', type=int, default=0,
                        help="""offset in memory loss""")
    parser.add_argument('--teacher_views', type=int, default=None,
                        help="""number of views to pass to teacher""")
    parser.add_argument('--local_first', type=utils.bool_flag, default=False,
                        help="""Whether to apply local transformation first in augmentation""")
    parser.add_argument('--return_pred_out', type=utils.bool_flag, default=False,
                        help="""Whether return prediction in teacher for eval""")

    return parser


def train_svt(args):
    # torch.autograd.set_detect_anomaly(True)

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    utils.restore_yt_checkpoint(args)
    utils.synchronize()

    print('args.world_size', args.world_size)
    args.batch_size = args.batch_size_per_gpu * args.world_size

    # ============ preparing data ... ============
    config = load_config(args)
    if utils.is_main_process():
        json.dump(vars(args), open(Path(args.output_dir) / "config.txt", "w"), indent=4)
    config.DATA.PATH_TO_DATA_DIR = args.data_path
    config.DATA.SAMPLING_RATE = args.sampling_rate
    config.DATA.NUM_FRAMES = args.num_frames
    config.local_crops_number = args.local_crops_number
    config.n_parts = args.n_parts
    config.n_global_views = args.n_global_views
    config.random_sampling = args.random_sampling
    config.global_size = args.global_size
    config.temporal_aug = (args.loss == "DINOLoss")
    config.global_crops_scale = args.global_crops_scale
    config.temporal_aug_memory = args.temporal_aug_memory
    config.local_first = args.local_first
    args.teacher_views = args.n_global_views if args.teacher_views is None else args.teacher_views

    # config.DATA.PATH_PREFIX = os.path.dirname(args.data_path)
    Dataset = datasets.__dict__[args.dataset]
    dataset = Dataset(cfg=config, mode="train", num_retries=10,
                      extension=args.video_extension, level=args.dataset_level,
                      pseudo_length=args.pseudo_length)
    if args.continuous:
        sampler = datasets.ContinuousBeg2EndHardSampler(dataset, batch_size=args.batch_size, world_size=args.world_size)
    else:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train data loaded: there are {len(dataset)} images.")

    def get_eval_datasets(eval_dataset, args):
        config.DATA.PATH_PREFIX = args.val_data_dir
        Eval_Dataset = datasets.__dict__[eval_dataset]
        print('Eval_Dataset', Eval_Dataset)
        eval_train = Eval_Dataset(cfg=config, mode="train", num_retries=10)
        eval_test = Eval_Dataset(cfg=config, mode="val", num_retries=10)

        sampler_train = torch.utils.data.DistributedSampler(eval_train, shuffle=False)
        eval_loader_train = torch.utils.data.DataLoader(
            eval_train, sampler=sampler_train, batch_size=args.eval_batch_size_per_gpu, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
        sampler_val = torch.utils.data.DistributedSampler(eval_test, shuffle=False)
        eval_loader_test = torch.utils.data.DataLoader(
            eval_test, sampler=sampler_val, batch_size=args.eval_batch_size_per_gpu, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
        print(f"Data loaded with {len(eval_train)} train and {len(eval_test)} val imgs.")
        return eval_train, eval_test, eval_loader_train, eval_loader_test

    if args.do_eval:
        # validation data
        config.DATA.PATH_TO_DATA_DIR = args.val_data_dir
        config.DATA.PATH_PREFIX = ""
        config.TEST.NUM_SPATIAL_CROPS = 1
        if args.eval_dataset:
            eval_train, eval_test, eval_loader_train, eval_loader_test = get_eval_datasets(args.eval_dataset, args)
        if args.eval_dataset2:
            eval_train2, eval_test2, eval_loader_train2, eval_loader_test2 = get_eval_datasets(args.eval_dataset2, args)

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch == "timesformer":
        Model = models.__dict__[args.model_name]
        print('Model', Model)
        student = Model(cfg=config, no_head=True, patch_size=args.patch_size, default_cfg=args.default_cfg)
        teacher = Model(cfg=config, no_head=True, patch_size=args.patch_size, default_cfg=args.default_cfg)
        embed_dim = student.embed_dim

        if args.pretrained_rgb is not None:
            state_dict = torch.load(args.pretrained_rgb)["teacher"]
            state_dict = {x[len("backbone."):]: y for x, y in state_dict.items() if x.startswith("backbone.")}
            msg = student.load_state_dict(state_dict)
            print(f"Loaded pretrained rgb student: {msg}")
            msg = teacher.load_state_dict(state_dict)
            print(f"Loaded pretrained rgb teacher: {msg}")
    if args.arch == "swin":
        student = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
        teacher = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])

        embed_dim = 1024
        print("Loaded swin transformer network")

    elif args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    Predictor = models.__dict__[args.predictor] if args.predictor else None
    Predictor_past = models.__dict__[args.predictor_past] if args.predictor_past else None
    HeadProba = models.__dict__[args.headproba] if args.headproba else None
    print('Predictor', Predictor)
    print('Predictor_past', Predictor_past)
    print('HeadProba', HeadProba)

    # multi-crop wrapper handles forward with inputs of different resolutions
    Wrapper = getattr(utils, args.wrapper)
    print('Wrapper', Wrapper)
    n_embd = embed_dim if args.wrapper in ['MultiCropWrapperPredictorProjector', 'MultiCropWrapperMemory', 'MultiCropWrapperMemorySaver'] else 256
    layer_norm = True if args.wrapper in ['MultiCropWrapperPredictorProjector', 'MultiCropWrapperMemory', 'MultiCropWrapperMemorySaver'] else False
    print('n_embd', n_embd)
    print('layer_norm', layer_norm)
    student = Wrapper(student,
         DINOHead(
             embed_dim,
             args.out_dim,
             use_bn=args.use_bn_in_head,
             norm_last_layer=args.norm_last_layer,
             skip_last=args.skip_last,
             bottleneck_dim=args.bottleneck_dim,
         ),
         predictor=Predictor(n_embd=n_embd, block_size=args.block_size, model_type=args.predictor_model_type,
                             layer_norm=layer_norm) if Predictor else None,
         predictor_past=Predictor_past(n_embd=n_embd, block_size=args.block_size, model_type=args.predictor_model_type,
                                       layer_norm=layer_norm) if Predictor_past else None,
         headprob=HeadProba(args.out_dim) if HeadProba else None,
         return_prediction_logits=args.return_prediction_logits,
         return_enc_logits=args.return_enc_logits,
         n_global_views=args.n_global_views,
         batch_size=args.batch_size_per_gpu,
         maxlen=args.maxlen,
         student=True,
         )
    teacher = Wrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head,
                 skip_last=args.skip_last,
                 bottleneck_dim=args.bottleneck_dim,),
        predictor=Predictor(n_embd=n_embd, block_size=args.block_size, model_type=args.predictor_model_type,
                            layer_norm=layer_norm) if Predictor else None,
        predictor_past=Predictor_past(n_embd=n_embd, block_size=args.block_size, model_type=args.predictor_model_type,
                                      layer_norm=layer_norm) if Predictor_past else None,
        headprob=HeadProba(args.out_dim) if HeadProba else None,
        return_prediction_logits=args.return_prediction_logits,
        return_enc_logits=args.return_enc_logits,
        n_global_views=args.n_global_views,
        batch_size=args.batch_size_per_gpu,
        maxlen=args.maxlen,
        student=False,
    )

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=False)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False)
    msg = teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    print(f"initialized teacher with student msg: {msg}")
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    Loss = losses.__dict__[args.loss]
    dino_loss = Loss(
        args.out_dim,
        ncrops=args.local_crops_number + args.n_global_views,  # total number of crops = 2 global crops + local_crops_number
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        n_global_views=args.n_global_views,
        local_crops_number=args.local_crops_number,
        args=args,
        start_video_idx=dataset._start_video_idx if args.continuous else None,
        video_clip_size=dataset._video_clip_size if args.continuous else None,
        index2clip_video=dataset.index2clip_video if args.continuous else None,
        batch_size=args.batch_size_per_gpu,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    if args.full_pretrain is not None:
        print('load pretrain!')
        utils.restart_from_pretrain(
            args.full_pretrain,
            student=student.module,
            teacher=teacher,
        )

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "step": 0}
    print('load restart!')
    args.wandb_id = utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch, start_step = to_restore["epoch"], to_restore["step"]

    if args.use_wandb and utils.is_main_process():
        wandb.init(
            project='causal_videov4',
            config=config,
            entity="avecplezir",
            reinit=True,
            # Restore parameters
            resume="allow",
            id=args.wandb_id,
            name=args.exp_name,
        )
        wandb.config.update(config, allow_val_change=True)
        wandb.run.log_code(".")

    step = start_step if start_step else start_epoch*len(dataset)
    print('start_step', start_step)
    print('step', step)

    if args.do_eval and args.do_eval_before_train:
        if args.eval_dataset:
            val_stats = eval_knn(eval_loader_train, eval_loader_test, eval_train, eval_test, teacher.backbone, opt=args)
        if args.eval_dataset2:
            val_stats2 = eval_knn(eval_loader_train2, eval_loader_test2, eval_train2, eval_test2, teacher.backbone,
                                  opt=args, return_pred_out=args.return_pred_out)
        if utils.is_main_process():
            print('val_stats', val_stats)
            print('val_stats mean', val_stats2)
            if args.use_wandb:
                wandb.log({'knn/' + key: value for key, value in val_stats.items()}, step=step)
                wandb.log({'knn/mean_' + key: value for key, value in val_stats2.items()}, step=step)
        utils.synchronize()

    start_time = time.time()
    print("Starting DINO training !")
    print('len(dataset)', len(dataset))
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats, step = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, step, fp16_scaler, args, cfg=config)

        # ============ eval ========================
        if args.do_eval and epoch % args.eval_freq == 0:
            val_stats = eval_knn(eval_loader_train, eval_loader_test, eval_train, eval_test, teacher.backbone, opt=args)
            val_stats2 = eval_knn(eval_loader_train2, eval_loader_test2, eval_train2, eval_test2, teacher.backbone,
                                  opt=args)
            if utils.is_main_process():
                print('val_stats', val_stats)
                print('val_stats mean', val_stats2)
                if args.use_wandb:
                    wandb.log({'knn/' + key: value for key, value in val_stats.items()}, step=step)
                    wandb.log({'knn/mean_'+key: value for key, value in val_stats2.items()}, step=step)
            utils.synchronize()

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'step': step,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        utils.save_checkpoint_to_yt(args, save_dict)
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            utils.save_checkpoint_to_yt(args, save_dict, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}  # **{f'val_{k}': v for k, v in val_stats.items()},
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.use_wandb:
                wandb.log({'epoch': epoch}, step=step)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, step,
                    fp16_scaler, args, cfg=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, indices, video_indices, *_) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if args.continuous:
            def all_unique(item):
                return len(set(item)) == len(item)
            assert all_unique(video_indices), 'videos in the batch are not unique!'
        # update step for wandb
        it = len(data_loader) * epoch + it  # global training iteration
        step += args.batch_size
        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        indices = indices.cuda(non_blocking=True)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:args.teacher_views], indices=indices,
                                     video_indices=video_indices)  # only the 2 global views pass through the teacher
            student_output = student(images, indices=indices, video_indices=video_indices)
            loss, dict_losses = dino_loss(student_output, teacher_output, epoch,
                                          student=student, teacher=teacher, video_indices=video_indices)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            print('dict_losses', dict_losses)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(**dict_losses)

        if it % args.log_every == 0 and utils.is_main_process() and args.use_wandb:
            wandb.log(dict(
                batch_loss=loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                wd=optimizer.param_groups[0]["weight_decay"],
                **{f"batch_{key}": val for key, val in dict_losses.items()}, step=step,
            ))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, step


def eval_knn(train_loader, test_loader, train_dataset, test_dataset, model, opt, return_pred_out=False):
    model.eval()  # teacher model already on eval
    print("Extracting features for train set...")
    train_features = extract_features(model, train_loader, return_pred_out=return_pred_out)
    print("Extracting features for val set...")
    test_features = extract_features(model, test_loader, return_pred_out=return_pred_out)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s for s in train_dataset._labels]).long()
    test_labels = torch.tensor([s for s in test_dataset._labels]).long()

    model.train()
    if utils.get_rank() == 0:
        train_features = train_features.cuda()
        test_features = test_features.cuda()
        train_labels = train_labels.cuda()
        test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        top1, top5 = knn_classifier(train_features, train_labels,
                                    test_features, test_labels, opt.nb_knn, opt.temperature)
        return {"knn_top1": top1, "knn_top5": top5}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SVT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_svt(args)
