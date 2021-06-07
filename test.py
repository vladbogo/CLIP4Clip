from dataloaders.rawvideo_util import RawVideoExtractor

import os

from pathlib import Path

rv = RawVideoExtractor(framerate=1, size=224)

assert Path('/scratch/shared/beegfs/vlad/clip4clip/msrvtt_data//MSRVTT_Videos/video710.mp4').exists()

print(os.path.exists('/scratch/shared/beegfs/vlad/clip4clip/msrvtt_data//MSRVTT_Videos/video710.mp4'))

raw_video_data = rv.get_video_data('/scratch/shared/beegfs/vlad/clip4clip/msrvtt_data//MSRVTT_Videos/video710.mp4')
print(len(raw_video_data['video'].shape))
print(raw_video_data['video'].shape)
