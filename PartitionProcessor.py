import shutil
import subprocess
import sys
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

from MatchProcess import KnowledgePointProcessor
from yolov5_2023.TextProcessor import TextProcessor


class PartitionPer:
    def __init__(self, video_name, segments):
        self.video_name = video_name
        self.video_file = f"video/{video_name}.mp4"
        # self.segments = [tuple(map(int, arg.split(":"))) for arg in segments]
        self.segments = segments
        self.target_dir = "res"

    def create_target_directory(self):
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
            os.mkdir(self.target_dir)
        else:
            os.mkdir(self.target_dir)

    def partition_video(self):
        for i, (start, end) in enumerate(self.segments):
            output_file = f"{self.target_dir}/video_{i}.mp4"
            try:
                ffmpeg_extract_subclip(
                    self.video_file, start, end, targetname=output_file
                )
            except subprocess.CalledProcessError as e:
                print(f"Error while extracting subclip: {e}")


if __name__ == "__main__":
    video_name = sys.argv[1] if len(sys.argv) > 1 else "3.6"

    text_processor = TextProcessor(video_name)
    text_processor.process_knowledge_points()

    processor = KnowledgePointProcessor(video_name)
    seg = processor.process_knowledge_points()
    processor.dict_convert_tuple()
    print("tp_res= {}".format(processor.tuple_kp_dicts_list))
    processor.save_csv_three("./kp_3.txt")
    processor.save_csv("./kp_5.txt")

    partation_processor = PartitionPer(video_name, seg)
    partation_processor.create_target_directory()
    partation_processor.partition_video()
