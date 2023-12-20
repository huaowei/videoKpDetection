import shutil
import subprocess
import sys

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

# 视频文件名
if len(sys.argv) < 2:
    print("Usage: python 4_partition.py <video_name> <segment1> <segment2> ...")
    sys.exit(1)

video_name = sys.argv[1]
# video_name ='01'
print(video_name)
video_file = 'video/' + video_name + '.mp4'

# 分割时间段（单位：秒）
segments = [tuple(map(int, arg.split(":"))) for arg in sys.argv[2:]]

print(segments)
# 定义目标目录
target_dir = 'res'

# 检查目标目录是否存在
if os.path.exists(target_dir):
    # 如果目录存在，清空目录下的内容
    shutil.rmtree(target_dir)
    os.mkdir(target_dir)
else:
    # 如果目录不存在，创建目录
    os.mkdir(target_dir)

# 分割视频并保存到res目录
for i, (start, end) in enumerate(segments):
    output_file = f'{target_dir}/video_{i}.mp4'
    try:
        ffmpeg_extract_subclip(video_file, start, end, targetname=output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error while extracting subclip: {e}")
