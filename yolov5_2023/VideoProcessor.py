# -*- coding: utf-8 -*-
from multiprocessing import Process
import os
import sys
import time
import cv2
import re
import unicodedata
from paddleocr import PaddleOCR
import torch
import detect_max

class VideoProcessor:
    def __init__(self, video_name, input_directory="./video", interval_seconds=10):
        self.video_name = video_name
        self.input_directory = input_directory
        self.interval_seconds = interval_seconds
        self.video_filename = f"{self.video_name}.mp4"
        self.result_folder = os.path.join("./result_all_txt", self.video_name)
        self.video_folder = os.path.join(self.input_directory, self.video_filename)
        self.yolo_folder = "./yolo_res"
        os.makedirs(self.result_folder, exist_ok=True)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.ocr = None
    def clean_filename(self, filename):
        cleaned_filename = re.sub(r"[^\w\-_.() ]", "", filename)
        cleaned_filename = (
            unicodedata.normalize("NFKD", cleaned_filename)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        cleaned_filename = cleaned_filename.replace(" ", "_")
        return cleaned_filename

    def rename_videos(self):
        mp4_files = [
            file for file in os.listdir(self.input_directory) if file.endswith(".mp4")
        ]

        for mp4_file in mp4_files:
            video_file = os.path.join(self.input_directory, mp4_file)
            cleaned_video_name = self.clean_filename(os.path.splitext(mp4_file)[0])
            new_video_file = os.path.join(
                self.input_directory, f"{cleaned_video_name}.mp4"
            )
            os.rename(video_file, new_video_file)

    def replace_punctuation_with_space(self, text):
        # 使用正则表达式将标点符号替换为空格
        new_text = re.sub(r"[^\w\s]", " ", text)
        # 使用正则表达式在数字和非数字之间插入空格
        new_text = re.sub(r"(\d+)(\D)", r"\1 \2", new_text)
        return new_text
    # 定义一个函数，用于运行 detect_max.run
    def run_detection(self,video_folder, yolo_folder, video_name, interval_frames):
        detect_max.run(
            source=video_folder,
            save_txt=True,
            save_conf=True,
            project=yolo_folder,
            name=video_name,
            device='0',
            vid_stride=interval_frames
        )
    def extract_frames(self, video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Unable to open video file: {video_file}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(fps)
        interval_frames = self.interval_seconds * fps
        # detect_max.run(source=self.video_folder,save_txt=True,save_conf=True,project=self.yolo_folder,name=self.video_name,device='0',vid_stride=interval_frames)
        # torch.cuda.empty_cache()
        # 创建一个进程并启动目标检测
        detect_process = Process(target=self.run_detection, args=(self.video_folder, self.yolo_folder, self.video_name, interval_frames))
        detect_process.start()
        time.sleep(10)
        # 终止进程
        detect_process.terminate()
        detect_process.join()  # 等待进程结束
        frame_count = 0
        image_count = 1
        self.ocr=PaddleOCR(use_angle_cls=True, lang="ch")


        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval_frames == 0 :
                print(frame_count)
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                cleaned_video_name = self.clean_filename(video_name)

                result = self.ocr.ocr(frame)

                merged_text = ""
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        if line[1][1] > 0.8:
                            new_txt = self.replace_punctuation_with_space(line[1][0])
                            merged_text += new_txt + "\n"
                            print(new_txt)

                txt_filename = f"{cleaned_video_name}_{image_count:03d}" + ".txt"
                txt_filepath = os.path.join(self.result_folder, txt_filename)

                with open(txt_filepath, "w", encoding="utf-8") as txt_file:
                    txt_file.write(merged_text)
                image_count += 1

            frame_count += 1

        cap.release()

    def process_all_mp4_files(self):
        self.rename_videos()
        mp4_files = [
            file for file in os.listdir(self.input_directory) if file.endswith(".mp4")
        ]

        for mp4_file in mp4_files:
            if mp4_file == self.video_filename:
                video_file = os.path.join(self.input_directory, mp4_file)
                self.extract_frames(video_file)


if __name__ == "__main__":
    video_name = sys.argv[1] if len(sys.argv) > 1 else "3.6"
    processor = VideoProcessor(video_name)
    processor.process_all_mp4_files()
