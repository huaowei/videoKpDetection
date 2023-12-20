# -*- coding: utf-8 -*-
import os
import shutil
import sys

import cv2
import re
import unicodedata
from skimage.metrics import structural_similarity as ssim


# video_name = sys.argv[1]
# print(video_name)
video_name = '4.1-4'
input_directory = "./train_video"
output_parent_directory = "./train_imgs"
interval_seconds = 10
video_filename = video_name + ".mp4"

def clean_filename(filename):
    # 去除非法字符并规范化字符串
    cleaned_filename = re.sub(r'[^\w\-_.() ]', '', filename)
    cleaned_filename = unicodedata.normalize('NFKD', cleaned_filename).encode('ascii', 'ignore').decode('utf-8')
    cleaned_filename = cleaned_filename.replace(' ', '_')
    return cleaned_filename
#
#
# def rename_videos(input_dir):
#     # 获取所有MP4文件
#     mp4_files = [file for file in os.listdir(input_dir) if file.endswith(".mp4")]
#
#     # 重命名每个MP4文件
#     for mp4_file in mp4_files:
#         video_file = os.path.join(input_dir, mp4_file)
#         cleaned_video_name = clean_filename(os.path.splitext(mp4_file)[0])
#         new_video_file = os.path.join(input_dir, f"{cleaned_video_name}.mp4")
#         os.rename(video_file, new_video_file)


def image_similarity(img1, img2):
    # Convert images to grayscale for SSIM calculation
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print(ssim(gray_img1, gray_img2))
    return ssim(gray_img1, gray_img2)


def extract_frames(video_file, output_dir, interval=10, similarity_threshold=1.0):
    # 如果目录存在，先清空目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"无法打开视频文件：{video_file}")
        return

    # 获取视频的帧率
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 设置间隔时间（以帧为单位）
    interval_frames = interval * fps

    # 初始化计数器
    frame_count = 0
    image_count = 1

    prev_frame = None

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔一定间隔保存一帧
        if frame_count % interval_frames == 0:
            # 检查当前帧与上一帧的相似度
            if prev_frame is not None and image_similarity(frame, prev_frame) >= similarity_threshold:
                print(f"Frame {frame_count} is similar to the previous frame. Skipping...")
            else:
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                cleaned_video_name = clean_filename(video_name)
                output_path = os.path.join(output_dir, f"{cleaned_video_name}_{image_count:03d}.jpg")
                print(output_path)
                cv2.imwrite(output_path, frame)
                image_count += 1

                # 保存当前帧用于下一次比较
                prev_frame = frame.copy()
        frame_count += 1

    # 释放视频对象
    cap.release()


def process_all_mp4_files(input_dir, output_parent_dir, interval=5, similarity_threshold=0.95):
    # 重命名视频文件
    # rename_videos(input_dir)

    # 获取所有MP4文件
    mp4_files = [file for file in os.listdir(input_dir) if file.endswith(".mp4")]

    # 处理每个MP4文件
    for mp4_file in mp4_files:
        if mp4_file == video_filename:
            print(mp4_file)
            video_file = os.path.join(input_dir, mp4_file)
            output_dir = os.path.join(output_parent_dir, os.path.splitext(mp4_file)[0])
            extract_frames(video_file, output_dir, interval, similarity_threshold)


process_all_mp4_files(input_directory, output_parent_directory, interval_seconds)

