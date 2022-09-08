# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch.utils.data
import numpy as np

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import glob
import warnings
import torch.utils.data
from torch.utils.data import Sampler

from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationEvents, VideoDataAugmentationDINO
from datasets.decoder import decode_events
from datasets.data_utils import get_random_sampling_rate
from datasets.epic_events import EpicEvents

from einops import rearrange


class EpicNEvents(EpicEvents):
    def __getitem__(self, index):
        for i_try in range(self._num_retries):
            try:
                ancor, video_idx = self.index2clip_video[index]
                left_limit = max(0, ancor-self.span)
                right_limit = min(ancor+self.span, self._video_clip_size[video_idx])
                clip_idx = np.random.choice(np.arange(left_limit, right_limit),
                                            size=self.cfg.n_global_views,
                                            replace=False)

                indices_sorted = sorted(clip_idx)

                frames = []
                for i in range(self.cfg.n_global_views):
                    frames.extend(self.get_event(indices_sorted[i], video_idx))

                # T H W C -> T C H W.
                frames = [rearrange(x, "t h w c -> t c h w") for x in frames]
                # Perform data augmentation.
                augmentation = VideoDataAugmentationEvents(size=self.cfg.global_size,
                                                           local_crops_number=self.cfg.local_crops_number,
                                                           global_crops_scale=self.cfg.global_crops_scale,
                                                           )
                frames = augmentation(frames, from_list=True, no_aug=self.cfg.DATA.NO_SPATIAL)
                # T C H W -> C T H W.
                frames = [rearrange(x, "t c h w -> c t h w") for x in frames]

                indices_sorted = np.array(indices_sorted) - left_limit
                return frames, indices_sorted, video_idx
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
                temporal_aug_memory=self.cfg.temporal_aug_memory
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


class EpicNFEvents(EpicNEvents):
      def __getitem__(self, index):
        for i_try in range(self._num_retries):
            try:
                clip_idx, video_idx = self.index2clip_video[index]
                frames = self.get_event(clip_idx, video_idx)

                # T H W C -> T C H W.
                frames = [rearrange(x, "t h w c -> t c h w") for x in frames]
                # Perform data augmentation.
                augmentation = VideoDataAugmentationDINO(size=self.cfg.global_size,
                                                         local_crops_number=self.cfg.local_crops_number,
                                                         global_crops_scale=self.cfg.global_crops_scale,
                                                         n_global_views=self.cfg.n_global_views,
                                                         )
                if self.cfg.temporal_aug_memory:
                    frames = augmentation(frames, from_list=True)
                else:
                    frames = augmentation(frames[0], from_list=False)
                # T C H W -> C T H W.
                frames = [rearrange(x, "t c h w -> c t h w") for x in frames]

                return frames, clip_idx, video_idx
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