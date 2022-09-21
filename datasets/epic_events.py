# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch.utils.data
import numpy as np

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import glob
import warnings
import torch.utils.data
from torch.utils.data import Sampler

from datasets.data_utils import get_random_sampling_rate
from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationEvents, VideoDataAugmentationDINO
from datasets.decoder import decode_events
from einops import rearrange


class EpicEvents(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, num_retries=10, extension='avi', level=1, **kwargs):
        # Only support train, val, and test mode.
        self.cfg = cfg
        self._num_retries = num_retries
        self.sampling_rate = get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        self.num_frames = self.cfg.DATA.NUM_FRAMES

        print("Constructing EpicEvents...")
        self._path_to_videos = glob.glob(self.cfg.DATA.PATH_TO_DATA_DIR + '/*' * level + '.' + extension)
        self.num_videos = len(self._path_to_videos )

        self._start_video_idx = [] # index with which video is started
        self._video_clip_size = [] # len of the video in terms of number of clips
        self.index2clip_video = {}
        self.init_video_clip_indices()

    def init_video_clip_indices(self, ):
        idx = 0
        for video_idx in range(len(self._path_to_videos)):
            container = get_video_container(
                self._path_to_videos[video_idx],
                self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                self.cfg.DATA.DECODING_BACKEND,
            )

            video_size = container.streams.video[0].frames
            fps = float(container.streams.video[0].average_rate)
            clip_size = self.sampling_rate * self.num_frames / self.cfg.DATA.TARGET_FPS * fps
            num_clips = max(int(video_size // clip_size), 1)
            self._start_video_idx.append(idx)
            self._video_clip_size.append(num_clips)
            for clip_idx in range(num_clips):
                self.index2clip_video[idx] = clip_idx, video_idx
                idx += 1

        if self._video_clip_size:
            vc = np.array(self._video_clip_size)
            print('video_clip_size stats, mean, std, max, min', vc.mean(), vc.std(), vc.max(), vc.min())

    def get_event(self, clip_idx, video_idx):
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]

        video_container = None
        try:
            video_container = get_video_container(
                self._path_to_videos[video_idx],
                self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                self.cfg.DATA.DECODING_BACKEND,
            )
        except Exception as e:
            print(
                "Failed to load video from {} with error {}".format(
                    self._path_to_videos[video_idx], e
                )
            )
        # Select a random video if the current video was not able to access.
        if video_container is None:
            warnings.warn(
                "Failed to meta load video idx {} from {}".format(
                    video_idx, self._path_to_videos[video_idx]
                )
            )
            return None

        # Decode video. Meta info is used to perform selective decoding.
        try:
            frames, indices = decode_events(
                container=video_container,
                sampling_rate=self.sampling_rate,
                num_frames=self.num_frames,
                clip_idx=clip_idx,
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                num_clips_global=self.cfg.n_global_views,
                n_parts=self.cfg.n_parts,
                random_sampling=self.cfg.random_sampling,
                mode='ordered',
                temporal_aug=self.cfg.temporal_aug,
                local_crops_number=self.cfg.local_crops_number,
            )
        except Exception as e:
            print(
                "Failed to decode events from video from {} with error {}".format(
                    self._path_to_videos[video_idx], e
                )
            )
            return None

        # If decoding failed (wrong format, video is too short, and etc),
        # select another video.
        if frames is None:
            warnings.warn(
                "Failed to decode video idx {} from {}".format(
                    video_idx, self._path_to_videos[video_idx]
                )
            )
            return None

        # augmentation
        frames = [rearrange(x, "t h w c -> t c h w") for x in frames]
        if self.cfg.temporal_aug:
            augmentation = VideoDataAugmentationDINO(size=self.cfg.global_size,
                                                     global_crops_scale=self.cfg.global_crops_scale,
                                                     local_crops_number=self.cfg.local_crops_number,
                                                     )
        else:
            augmentation = VideoDataAugmentationEvents(size=self.cfg.global_size,
                                                       local_crops_number=self.cfg.local_crops_number,
                                                       global_crops_scale=self.cfg.global_crops_scale,
                                                       local_first=self.cfg.local_first,
                                                       )
        frames = augmentation(frames, from_list=True, no_aug=self.cfg.DATA.NO_SPATIAL)
        frames = [rearrange(x, "t c h w -> c t h w") for x in frames]

        return frames, indices

    def __getitem__(self, index):
        for i_try in range(self._num_retries):
            try:
                clip_idx, video_idx = self.index2clip_video[index]
                frames, indices = self.get_event(clip_idx, video_idx)
                return frames, indices, clip_idx, video_idx
            except:
                if i_try > self._num_retries // 2:
                    # let's try another one
                    index = index + 1
                    continue

        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return sum(self._video_clip_size)


class ContinuousSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.epoch = 0

    def __iter__(self):
        iters = [
            iter(range(self.data_source._start_video_idx[i],
                       self.data_source._start_video_idx[i] + self.data_source._video_clip_size[i]))
            for i in range(self.data_source.num_videos)
        ]

        while True:
            try:
                for video_idx, itr in enumerate(iters):
                    yield next(itr)
            except StopIteration:
                print(f'StopIteration, redefining iterator for {video_idx} video')
                iters[video_idx] = iter(range(self.data_source._start_video_idx[video_idx],
                                              self.data_source._start_video_idx[video_idx] +
                                              self.data_source._video_clip_size[video_idx]))
                continue

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return sum(self.data_source._video_clip_size)

    def set_epoch(self, epoch):
        self.epoch = epoch


class ContinuousRandomSampler(ContinuousSampler):
    def __init__(self, data_source, batch_size=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.epoch = 0

    def __iter__(self):
        iters = [
            {i: iter(range(self.data_source._start_video_idx[i],
                           self.data_source._start_video_idx[i] + self.data_source._video_clip_size[i]))}
            for i in range(self.data_source.num_videos)
        ]

        p = np.array([self.data_source._video_clip_size[i] for i in range(self.data_source.num_videos)])
        p = p / p.sum()

        choices = np.random.choice(iters, size=self.batch_size, p=p, replace=False)
        iteration = 0
        while True:
            if iteration % self.batch_size == 0:
                choices = np.random.choice(iters, size=self.batch_size, p=p, replace=False)
            iteration += 1
            for choice in choices:
                video_idx, itr = tuple(choice.items())[0]
                try:
                    yield next(itr)
                except StopIteration:
                    print(f'StopIteration, redefining iterator for {video_idx} video')
                    iters[video_idx] = {video_idx: iter(range(self.data_source._start_video_idx[video_idx],
                                                              self.data_source._start_video_idx[video_idx] +
                                                              self.data_source._video_clip_size[video_idx]))}
                    video_idx, itr = tuple(iters[video_idx].items())[0]
                    yield next(itr)


class ContinuousBeg2EndSampler(ContinuousSampler):
    def __init__(self, data_source, batch_size=None, world_size=None, verbose=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.epoch = 0
        self.world_size = world_size
        self.verbose = verbose

    def __iter__(self):
        iters = [iter(range(self.data_source._start_video_idx[i],
                           self.data_source._start_video_idx[i] + self.data_source._video_clip_size[i]))
            for i in range(self.data_source.num_videos)
        ]

        current_choices = np.random.choice(np.arange(len(iters)), size=self.batch_size, replace=False)
        full_indices = set(np.arange(len(iters)))
        current_choices = set(current_choices)
        offset = full_indices.difference(current_choices)
        while True:
            # print('video_indices', current_choices)
            for video_idx in current_choices:
                try:
                    yield next(iters[video_idx])
                except StopIteration:
                    if self.verbose:
                        print(f'StopIteration, redefining iterator for {video_idx} video')
                    iters[video_idx] = iter(range(self.data_source._start_video_idx[video_idx],
                                                              self.data_source._start_video_idx[video_idx] +
                                                              self.data_source._video_clip_size[video_idx]))
                    current_choices.remove(video_idx)
                    offset.add(video_idx)
                    replacement_idx = np.random.choice(list(offset), replace=False)
                    current_choices.add(replacement_idx)
                    offset.remove(replacement_idx)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return int(sum(self.data_source._video_clip_size) // self.world_size)


class ContinuousBeg2EndHardSampler(ContinuousBeg2EndSampler):
    def __iter__(self):
        iters = [iter(range(self.data_source._start_video_idx[i],
                            self.data_source._start_video_idx[i] + self.data_source._video_clip_size[i]))
                 for i in range(self.data_source.num_videos)
                 ]

        current_choices = np.random.choice(np.arange(len(iters)), size=self.batch_size, replace=False)
        full_indices = set(np.arange(len(iters)))
        current_choices = set(current_choices)
        current_choices_np = np.array(list(current_choices))
        offset = full_indices.difference(current_choices)
        sampler_idx = 0
        while True:
            for idx, video_idx in enumerate(current_choices_np):
                try:
                    yield next(iters[video_idx])
                except StopIteration:
                    if self.verbose:
                        print(f'StopIteration, redefining iterator for {video_idx} video')
                    iters[video_idx] = iter(range(self.data_source._start_video_idx[video_idx],
                                                  self.data_source._start_video_idx[video_idx] +
                                                  self.data_source._video_clip_size[video_idx]))
                    current_choices.remove(video_idx)
                    offset.add(video_idx)
                    replacement_idx = np.random.choice(list(offset), replace=False)
                    current_choices.add(replacement_idx)
                    offset.remove(replacement_idx)
                    current_choices_np[idx] = replacement_idx
                    yield next(iters[replacement_idx])
                sampler_idx += 1

            if sampler_idx > len(self):
                break