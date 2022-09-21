# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from datasets.epic_events import EpicEvents


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


class EpicBertEvents(EpicEvents):
    def __getitem__(self, index):
        for i_try in range(self._num_retries):
            try:
                ancor, video_idx = self.index2clip_video[index]
                left_limit = max(0, ancor-self.cfg.block_size)
                right_limit = ancor
                clip_idx = np.random.choice(np.arange(left_limit, right_limit),
                                            size=self.cfg.n_sampled_parts-1,
                                            replace=False)

                clip_idx = list(np.array(sorted(clip_idx)))
                indices_sorted = clip_idx + [ancor]

                frames = []
                for i in range(self.cfg.n_sampled_parts):
                    frames.extend(self.get_event(indices_sorted[i], video_idx))

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