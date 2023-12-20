# -*- coding: utf-8 -*-
import os
import sys
import cv2
import re
import unicodedata
from paddleocr import PaddleOCR


class VideoProcessor:
    def __init__(self, video_name, input_directory="./video", interval_seconds=10):
        self.video_name = video_name
        self.input_directory = input_directory
        self.interval_seconds = interval_seconds
        self.video_filename = f"{self.video_name}.mp4"
        self.result_folder = os.path.join('./result_all_txt', self.video_name)
        os.makedirs(self.result_folder, exist_ok=True)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")

    def clean_filename(self, filename):
        cleaned_filename = re.sub(r'[^\w\-_.() ]', '', filename)
        cleaned_filename = unicodedata.normalize('NFKD', cleaned_filename).encode('ascii', 'ignore').decode('utf-8')
        cleaned_filename = cleaned_filename.replace(' ', '_')
        return cleaned_filename

    def rename_videos(self):
        mp4_files = [file for file in os.listdir(self.input_directory) if file.endswith(".mp4")]

        for mp4_file in mp4_files:
            video_file = os.path.join(self.input_directory, mp4_file)
            cleaned_video_name = self.clean_filename(os.path.splitext(mp4_file)[0])
            new_video_file = os.path.join(self.input_directory, f"{cleaned_video_name}.mp4")
            os.rename(video_file, new_video_file)

    def replace_punctuation_with_space(self, text):
        # 使用正则表达式将标点符号替换为空格
        new_text = re.sub(r'[^\w\s]', ' ', text)
        # 使用正则表达式在数字和非数字之间插入空格
        new_text = re.sub(r'(\d+)(\D)', r'\1 \2', new_text)
        return new_text

    def extract_frames(self, video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Unable to open video file: {video_file}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        interval_frames = self.interval_seconds * fps
        frame_count = 0
        image_count = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval_frames == 0:
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

                txt_filename = f"{cleaned_video_name}_{image_count:03d}" + '.txt'
                txt_filepath = os.path.join(self.result_folder, txt_filename)

                with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(merged_text)
                image_count += 1

            frame_count += 1

        cap.release()

    def process_all_mp4_files(self):
        self.rename_videos()
        mp4_files = [file for file in os.listdir(self.input_directory) if file.endswith(".mp4")]

        for mp4_file in mp4_files:
            if mp4_file == self.video_filename:
                video_file = os.path.join(self.input_directory, mp4_file)
                self.extract_frames(video_file)

if __name__ == "__main__":
    video_name = sys.argv[1] if len(sys.argv) > 1 else '3.6'
    processor = VideoProcessor(video_name)
    processor.process_all_mp4_files()
