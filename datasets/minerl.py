import minerl
import random
import numpy as np
import torch

from einops import rearrange

from datasets.transform import VideoDataAugmentationEvents


class MineRL(torch.utils.data.Dataset):

    def __init__(self, name='MineRLNavigate-v0', num_frames=8, num_clips=4): #MineRLNavigate-v0 MineRLObtainDiamond-v0
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.data = minerl.data.make(name, data_dir='/home/ivananokhin/mineRL/')
        self.trajectory_names = self.data.get_trajectory_names()
        random.shuffle(self.trajectory_names)
        self.trajectories = {trajectory_name: self.data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
                             for trajectory_name in self.trajectory_names}
        print('done init')

    def __getitem__(self, index):
        trajectory_name = self.trajectory_names[index]
        out = []
        start_new_episode = False
        for clip in range(self.num_clips):
            all_pov_obs = []
            for frame in range(self.num_frames):
                try:
                    obs, *_ = next(self.trajectories[trajectory_name])
                except StopIteration:
                    print(f'create new {trajectory_name} generator')
                    start_new_episode = True
                    self.trajectories[trajectory_name] = self.data.load_data(trajectory_name, skip_interval=0,
                                                                             include_metadata=False)
                    obs, *_ = next(self.trajectories[trajectory_name])
                all_pov_obs.append(obs["pov"])
            out.append(np.array(all_pov_obs))

        # Perform data augmentation.
        out = [rearrange(x, "t h w c -> t c h w") for x in out]
        augmentation = VideoDataAugmentationEvents(size=64)
        out = augmentation(out, from_list=True, no_aug=True)
        out = [rearrange(x, "t c h w -> c t h w") for x in out]

        return out, index, start_new_episode

    def __len__(self):
        return len(self.trajectory_names)