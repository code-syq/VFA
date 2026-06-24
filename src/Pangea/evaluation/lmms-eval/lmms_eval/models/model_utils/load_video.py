import av
from av.codec.context import CodecContext
import numpy as np


def load_video_decord(video_path, max_frames_num):
    """
    Load (up to) `max_frames_num` frames uniformly from a video using Decord.

    Note: Some models in this repo expect this helper (it exists in the upstream
    lmms-eval fork under `src/lmms-eval/...`). Pangea's copy previously missed it,
    which caused import-time failures for video-capable models (e.g. qwen2_vl).
    """
    from decord import VideoReader, cpu

    if isinstance(video_path, str):
        path = video_path
    else:
        # Some callers pass [path] or (path, ...)
        path = video_path[0]

    vr = VideoReader(path, ctx=cpu(0))
    total_frame_num = len(vr)
    if total_frame_num <= 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    # Use uniform sampling; duplicates are OK when total_frame_num < max_frames_num.
    frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames  # (frames, height, width, channels)


# This one is faster
def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return frames


# This one works for all types of video
def record_video_length_packet(container):
    frames = []
    # https://github.com/PyAV-Org/PyAV/issues/1269
    # https://www.cnblogs.com/beyond-tester/p/17641872.html
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames


def read_video_pyav(video_path, num_frm=8):
    container = av.open(video_path)

    if "webm" not in video_path and "mkv" not in video_path:
        # For mp4, we try loading with stream first
        try:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            sampled_frm = min(total_frames, num_frm)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            frames = record_video_length_stream(container, indices)
        except:
            container = av.open(video_path)
            frames = record_video_length_packet(container)
            total_frames = len(frames)
            sampled_frm = min(total_frames, num_frm)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            frames = [frames[i] for i in indices]
    else:
        container = av.open(video_path)
        frames = record_video_length_packet(container)
        total_frames = len(frames)
        sampled_frm = min(total_frames, num_frm)
        indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
        frames = [frames[i] for i in indices]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])
