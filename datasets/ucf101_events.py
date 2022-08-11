import os
import random
import warnings

import torch.utils.data

from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
from datasets.decoder import decode, decode_events
from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationEvents

from einops import rearrange


class UCF101Events(torch.utils.data.Dataset):
    """
    UCF101 video loader. Construct the UCF101 video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the UCF101 video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train mode, the data loader will take data from the
                train set, and sample one clip per video. For the val and
                test mode, the data loader will take data from relevent set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test"], "Split '{}' not supported for UCF101".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        self._split_idx = mode
        # For training mode, one single clip is sampled from every video. For validation or testing, NUM_ENSEMBLE_VIEWS
        # clips are sampled from every video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from the frames.
        self._num_clips = 1

        print("Constructing UCF101 {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "ucf101_{}_split_1_videos.txt".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                        len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                        == 2
                )
                name, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                path = self.cfg.DATA.PATH_TO_DATA_DIR + '/' + name
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (len(self._path_to_videos) > 0), f"Failed to load UCF101 split {self._split_idx} from {path_to_file}"
        print(f"Constructing UCF101 dataloader (size: {len(self._path_to_videos)}) from {path_to_file}")

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        min_scale, max_scale, crop_size = (
            [self.cfg.DATA.TEST_CROP_SIZE] * 3 if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
            else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2 + [self.cfg.DATA.TEST_CROP_SIZE]
        )
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale}) == 1

        sampling_rate = get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatedly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                print(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                warnings.warn(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )

            # Decode video. Meta info is used to perform selective decoding.
            frames = decode_events(
                container=video_container,
                sampling_rate=sampling_rate,
                num_frames=self.cfg.DATA.NUM_FRAMES,
                clip_idx=-1,
                num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                num_clips_2=self.cfg.local_crops_number + self.cfg.n_global_views,
                n_parts=self.cfg.n_parts,
                random_sampling=self.cfg.random_sampling
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                warnings.warn(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[index]

            # T H W C -> T C H W.
            frames = [rearrange(x, "t h w c -> t c h w") for x in frames]

            # Perform data augmentation.
            augmentation = VideoDataAugmentationEvents()
            frames = augmentation(frames, from_list=True, no_aug=True)

            # T C H W -> C T H W.
            frames = [rearrange(x, "t c h w -> c t h w") for x in frames]

            # Perform temporal sampling from the fast pathway.
            frames = [torch.index_select(
                x,
                1,
                torch.linspace(
                    0, x.shape[1] - 1, x.shape[1] if self.cfg.DATA.RAND_FR else self.cfg.DATA.NUM_FRAMES

                ).long(),
            ) for x in frames]

            return frames, label, index, {}
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
        return len(self._path_to_videos)


class UCFEventsReturnIndexDataset(UCF101Events):
    def __getitem__(self, idx):
        img, _, _, _ = super(UCFEventsReturnIndexDataset, self).__getitem__(idx)
        return img, idx