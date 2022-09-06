# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch.utils.data
import numpy as np

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import glob
import warnings
import torch.utils.data
from torch.utils.data import Sampler

from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationEvents
from datasets.decoder import decode_events
from datasets.data_utils import get_random_sampling_rate

from einops import rearrange


class EpicNEvents(torch.utils.data.Dataset):
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
            num_clips = int(video_size // clip_size)
            self._start_video_idx.append(idx)
            self._video_clip_size.append(num_clips)
            for clip_idx in range(num_clips):
                self.index2clip_video[idx] = clip_idx, video_idx
                idx += 1

    def __getitem__(self, index):

        for i_try in range(self._num_retries):
            try:
                _, video_idx = self.index2clip_video[index]
                print('self._video_clip_size[video_idx]', self._video_clip_size[video_idx])
                clip_idx = np.random.choice(np.arange(self._video_clip_size[video_idx]),
                                             size=self.cfg.n_global_views,
                                             replace=False)

                print('clip_idx', clip_idx)
                indices_sorted = sorted(clip_idx)
                print('indices_sorted', indices_sorted)

                frames = []
                for i in range(self.cfg.n_global_views):
                    print('indices_sorted[i]', indices_sorted[i])
                    frames.extend(self.get_event(indices_sorted[i], video_idx))

                print('frames', len(frames))
            except:
                if i_try > self._num_retries // 2:
                    # let's try another one
                    index = index + 1
                    continue

            # T H W C -> T C H W.
            frames = [rearrange(x, "t h w c -> t c h w") for x in frames]
            # Perform data augmentation.
            augmentation = VideoDataAugmentationEvents(local_crops_number=self.cfg.local_crops_number)
            frames = augmentation(frames, from_list=True, no_aug=self.cfg.DATA.NO_SPATIAL)
            # T C H W -> C T H W.
            frames = [rearrange(x, "t c h w -> c t h w") for x in frames]

            return frames, indices_sorted, video_idx

        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

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
                num_clips_global=1,
                n_parts=self.cfg.n_parts,
                random_sampling=self.cfg.random_sampling,
                mode='ordered',
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

        return frames

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


class ContinuousRandomSampler(Sampler):
    def __init__(self, data_source, batch_size=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

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