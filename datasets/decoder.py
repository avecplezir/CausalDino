# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import random
import torch
import torchvision.io as io


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def pyav_decode_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


def torchvision_decode(
    video_handle,
    sampling_rate,
    num_frames,
    clip_idx,
    video_meta,
    num_clips=10,
    target_fps=30,
    modalities=("visual",),
    max_spatial_scale=0,
):
    """
    If video_meta is not empty, perform temporal selective decoding to sample a
    clip from the video with TorchVision decoder. If video_meta is empty, decode
    the entire video and update the video_meta.
    Args:
        video_handle (bytes): raw bytes of the video file.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the clip_idx-th video clip.
        video_meta (dict): a dict contains VideoMetaData. Details can be found
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps.
        modalities (tuple): tuple of modalities to decode. Currently only
            support `visual`, planning to support `acoustic` soon.
        max_spatial_scale (int): the maximal resolution of the spatial shorter
            edge size during decoding.
    Returns:
        frames (tensor): decoded frames from the video.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): if True, the entire video was decoded.
    """
    # Convert the bytes to a tensor.
    video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))

    decode_all_video = True
    video_start_pts, video_end_pts = 0, -1
    # The video_meta is empty, fetch the meta data from the raw video.
    if len(video_meta) == 0:
        # Tracking the meta info for selective decoding in the future.
        meta = io._probe_video_from_memory(video_tensor)
        # Using the information from video_meta to perform selective decoding.
        video_meta["video_timebase"] = meta.video_timebase
        video_meta["video_numerator"] = meta.video_timebase.numerator
        video_meta["video_denominator"] = meta.video_timebase.denominator
        video_meta["has_video"] = meta.has_video
        video_meta["video_duration"] = meta.video_duration
        video_meta["video_fps"] = meta.video_fps
        video_meta["audio_timebas"] = meta.audio_timebase
        video_meta["audio_numerator"] = meta.audio_timebase.numerator
        video_meta["audio_denominator"] = meta.audio_timebase.denominator
        video_meta["has_audio"] = meta.has_audio
        video_meta["audio_duration"] = meta.audio_duration
        video_meta["audio_sample_rate"] = meta.audio_sample_rate

    fps = video_meta["video_fps"]
    if (
        video_meta["has_video"]
        and video_meta["video_denominator"] > 0
        and video_meta["video_duration"] > 0
    ):
        # try selective decoding.
        decode_all_video = False
        clip_size = sampling_rate * num_frames / target_fps * fps
        start_idx, end_idx = get_start_end_idx(
            fps * video_meta["video_duration"], clip_size, clip_idx, num_clips
        )
        # Convert frame index to pts.
        pts_per_frame = video_meta["video_denominator"] / fps
        video_start_pts = int(start_idx * pts_per_frame)
        video_end_pts = int(end_idx * pts_per_frame)

    # Decode the raw video with the tv decoder.
    v_frames, _ = io._read_video_from_memory(
        video_tensor,
        seek_frame_margin=1.0,
        read_video_stream="visual" in modalities,
        video_width=0,
        video_height=0,
        video_min_dimension=max_spatial_scale,
        video_pts_range=(video_start_pts, video_end_pts),
        video_timebase_numerator=video_meta["video_numerator"],
        video_timebase_denominator=video_meta["video_denominator"],
    )

    if v_frames.shape == torch.Size([0]):
        # failed selective decoding
        decode_all_video = True
        video_start_pts, video_end_pts = 0, -1
        v_frames, _ = io._read_video_from_memory(
            video_tensor,
            seek_frame_margin=1.0,
            read_video_stream="visual" in modalities,
            video_width=0,
            video_height=0,
            video_min_dimension=max_spatial_scale,
            video_pts_range=(video_start_pts, video_end_pts),
            video_timebase_numerator=video_meta["video_numerator"],
            video_timebase_denominator=video_meta["video_denominator"],
        )

    return v_frames, fps, decode_all_video


def pyav_decode(
    container, sampling_rate, num_frames, clip_idx, num_clips=10, target_fps=30, start=None, end=None
, duration=None, frames_length=None):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Try to fetch the decoding information from the video head. Some of the
    # videos does not support fetching the decoding information, for that case
    # it will get None duration.
    fps = float(container.streams.video[0].average_rate)

    orig_duration = duration
    tb = float(container.streams.video[0].time_base)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration
    if duration is None and orig_duration is not None:
       duration = orig_duration / tb

    if duration is None:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        start_idx, end_idx = get_start_end_idx(
            frames_length,
            sampling_rate * num_frames / target_fps * fps,
            clip_idx,
            num_clips,
        )
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    if start is not None and end is not None:
        decode_all_video = False

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        if start is None and end is None:
            video_frames, max_pts = pyav_decode_stream(
                container,
                video_start_pts,
                video_end_pts,
                container.streams.video[0],
                {"video": 0},
            )
        else:
            timebase = duration / frames_length
            start_i = start
            end_i = end
            video_frames, max_pts = pyav_decode_stream(
                container,
                start_i,
                end_i,
                container.streams.video[0],
                {"video": 0},
            )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))

    return frames, fps, decode_all_video


def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    backend="pyav",
    max_spatial_scale=0,
    start=None,
    end=None,
    duration=None,
    frames_length=None,
    temporal_aug=False,
    two_token=False,
    rand_fr=False,
    local_crops_number=8,
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
        temporal_aug: bool
        two_token: bool
        rand_fr: bool
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    try:
        if backend == "pyav":
            frames, fps, decode_all_video = pyav_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips,
                target_fps,
                start,
                end,
                duration,
                frames_length,
            )
        elif backend == "torchvision":
            frames, fps, decode_all_video = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                video_meta,
                num_clips,
                target_fps,
                ("visual",),
                max_spatial_scale,
            )
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None

    # Return None if the frames was not decoded successfully.
    if frames is None or frames.size(0) == 0:
        return None

    clip_sz = sampling_rate * num_frames / target_fps * fps
    start_idx, end_idx = get_start_end_idx(
        video_size=frames.shape[0],
        clip_size=clip_sz,
        clip_idx=clip_idx if decode_all_video else 0,
        num_clips=num_clips if decode_all_video else 1,
    )
    # Perform temporal sampling from the decoded video.
    if two_token:
        max_len = frames.shape[0]
        global_samples = []
        for _ in range(3):
            random_idx = random.randint(0, 6)
            cur_global = temporal_sampling(frames, random_idx, max_len - random_idx, num_frames)
            global_samples.append(cur_global)
        local_samples = []
        local_width = max_len // 8
        for _ in range(2):
            random_idx = random.randint(0, max_len - local_width - 1)
            cur_local = temporal_sampling(frames, random_idx, random_idx + local_width, num_frames)
            local_samples.append(cur_local)
        frames = [*global_samples, *local_samples]
    elif temporal_aug:
        max_len = frames.shape[0]

        if rand_fr:
            global_1 = temporal_sampling(frames, 0, max_len - 5, 4)
            global_2 = temporal_sampling(frames, 5, max_len, 8)
            local_samples = []
            local_width = max_len // 8
            num_local_frames = [2, 2, 4, 4, 8, 8, 16, 16]
            for l_idx in range(8):
                random_idx = random.randint(0, max_len - local_width - 1)
                cur_local = temporal_sampling(frames, random_idx, random_idx + local_width, num_local_frames[l_idx])
                local_samples.append(cur_local)
        else:
            num_global_frames = num_frames
            num_local_frames = num_frames
            global_1 = temporal_sampling(frames, 0, max_len - 5, num_global_frames)
            global_2 = temporal_sampling(frames, 5, max_len, num_global_frames)
            local_samples = []
            local_width = max_len // 8
            for _ in range(local_crops_number):
                random_idx = random.randint(0, max_len - local_width - 1)
                cur_local = temporal_sampling(frames, random_idx, random_idx + local_width, num_local_frames)
                local_samples.append(cur_local)

        frames = [global_1, global_2, *local_samples]

    else:
        frames = temporal_sampling(frames, start_idx, end_idx, num_frames)  # frames.shape = (T, H, W, C)
    return frames


def decode_events(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    backend="pyav",
    max_spatial_scale=0,
    start=None,
    end=None,
    duration=None,
    frames_length=None,
    temporal_aug=False,
    two_token=False,
    rand_fr=False,
    num_clips_2=2,
    random_sample=True,
    n_parts=9,
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
        temporal_aug: bool
        two_token: bool
        rand_fr: bool
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    try:
        if backend == "pyav":
            frames, fps, decode_all_video = pyav_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips,
                target_fps,
                start,
                end,
                duration,
                frames_length,
            )
        elif backend == "torchvision":
            frames, fps, decode_all_video = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                video_meta,
                num_clips,
                target_fps,
                ("visual",),
                max_spatial_scale,
            )
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None

    # Return None if the frames was not decoded successfully.
    if frames is None or frames.size(0) == 0:
        return None

    clip_sz = sampling_rate * num_frames / target_fps * fps
    start_idx, end_idx = get_start_end_idx(
        video_size=frames.shape[0],
        clip_size=clip_sz,
        clip_idx=clip_idx if decode_all_video else 0,
        num_clips=num_clips if decode_all_video else 1,
    )
    # Perform temporal sampling from the decoded video.
    if random_sample:
        max_len = frames.shape[0]
        samples = []
        local_width = max_len // n_parts
        start_idx = random.randint(0, local_width - 1)
        indices = np.random.choice(np.arange(0, n_parts-1), replace=False, size=num_clips_2)
        indices = sorted(indices)
        for idx in indices:
            cur_local = temporal_sampling(frames, start_idx+idx*local_width, start_idx + idx*local_width+local_width, num_frames)
            samples.append(cur_local)

        frames = [*samples]
    elif temporal_aug:
        max_len = frames.shape[0]

        samples = []
        local_width = max_len // (num_clips_2 + 1)
        idx = random.randint(0, local_width - 1)
        for _ in range(num_clips_2):
            cur_local = temporal_sampling(frames, idx, idx + local_width, num_frames)
            samples.append(cur_local)
            idx += local_width

        frames = [*samples]
    else:
        frames = temporal_sampling(frames, start_idx, end_idx, num_frames)  # frames.shape = (T, H, W, C)
    return frames
