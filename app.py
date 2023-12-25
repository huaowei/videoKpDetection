import json
import os
import shutil

import unicodedata
from flask import Flask, request, jsonify, send_file, send_from_directory
import subprocess
import re
import requests
from flask_cors import CORS

from yolov5_2023.VideoProcessor import *
from yolov5_2023.TextProcessor import *
from matchProcess import *

# 允许上传的文件扩展名的集合
ALLOWED_EXTENSIONS = {"txt", "xlx", "xlsx", "csv", "mp4"}
# 返回三元组文件
DOWNLOAD_KP_FILE = r"./kp_3.txt"
# 返回五元组文件
DOWNLOAD_TUPLE_FILE = r"./kp_5.txt"
app = Flask(__name__)
# 设置上传文件目录
app.config["UPLOAD_VIDEO_FOLDER"] = "./video"
app.config["UPLOAD_TRAIN_VIDEO_FOLDER"] = "./train_video"
app.config["UPLOAD_KP_FOLDER"] = "./kbase"
# 设置下载目录
app.config["DOWNLOAD_TRAIN_IMG_FILE"] = "./train_imgs"
app.config["DOWNLOAD_KP_FILE"] = DOWNLOAD_KP_FILE
app.config["DOWNLOAD_TUPLE_FILE"] = DOWNLOAD_TUPLE_FILE
# 解决跨域问题
cors = CORS()
cors.init_app(app, resource={r"/*": {"origins": "*"}})
# res_list_text = None
# res_tp_text = None
@app.route("/process", methods=["POST"])
def process_video():
    try:
        get_data = request.args.to_dict()
        # Get the 'video_name' parameter from the request
        video_name = get_data.get("video_name")
        # Validate the 'video_name' parameter here if needed
        # Your existing code for program execution
        programs = ["0+1.py", "2_detect_max_gap.py", "3_match_sj.py", "4_partition.py"]
        segments = None

        for program in programs:
            if program == "3_match_sj.py":
                output = subprocess.check_output(
                    ["python", program, video_name], encoding="gbk"
                )
                # 编写正则表达式模式来匹配需要的内容
                segments_match = re.search(r"key_list\s*=\s*\[(.*?)\]", output)
                res_match = re.search(r"res_list\s*=\s*\[(.*?)\)\]", output)
                tp_match = re.search(r"tp_res\s*=\s*\[(.*?)\}\]", output)
                if segments_match:
                    segments_res = eval(segments_match.group(1))
                    segments = segments_res
                if res_match:
                    res_list_text = "[" + res_match.group(1) + ")]"
                if tp_match:
                    res_tp_text = "[" + tp_match.group(1) + "}]"
            elif program == "4_partition.py":
                if segments is not None:
                    subprocess.run(
                        ["python", program, video_name]
                        + [f"{start}:{end}" for start, end in segments],
                        check=True,
                    )
                else:
                    return jsonify(
                        {
                            "error": "segments is not defined. Make sure to run 3_match_sj.py first."
                        }
                    )
            else:
                subprocess.run(["python", program, video_name], check=True)

        return jsonify(
            {
                "message": "All programs executed successfully",
                "segments": segments,
                "res_list_text": res_list_text,
                "res_tp_text": res_tp_text,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)})


# 上传知识库文件的接口
@app.route("/kp-abstract/upload/kp", methods=["POST"])
def upload_kp():
    if not os.path.exists(app.config["UPLOAD_KP_FOLDER"]):
        os.makedirs(app.config["UPLOAD_KP_FOLDER"])
    # 每次上传清空之前的文件
    for f in os.listdir(app.config["UPLOAD_KP_FOLDER"]):
        os.remove(os.path.join(app.config["UPLOAD_KP_FOLDER"], f))
    file = request.files["file"]
    fname = file.filename
    if allowed_file(fname):
        file.save(os.path.join(app.config["UPLOAD_KP_FOLDER"], fname))
    return "OK!"


# 检查上传文件的扩展名是否合法
def allowed_file(filename):
    # rsplit从优向左开始分割，第二个参数代表分割次数
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/kp-abstract/clear/kp", methods=["post"])
def clear_kp_dir():
    for f in os.listdir(app.config["UPLOAD_VIDEO_FOLDER"]):
        os.remove(os.path.join(app.config["UPLOAD_VIDEO_FOLDER"], f))
    return "OK!"


# 上传视频mp4文件的接口
@app.route("/kp-abstract/upload/mp4", methods=["POST"])
def upload_mp4():
    # 判断mp4视频文件是否为空
    if not os.path.exists(app.config["UPLOAD_VIDEO_FOLDER"]):
        os.makedirs(app.config["UPLOAD_VIDEO_FOLDER"])
    file = request.files["file"]
    fname = file.filename
    if allowed_file(fname):
        file.save(os.path.join(app.config["UPLOAD_VIDEO_FOLDER"], fname))
    return "OK!"


def clean_filename(filename):
    # 去除非法字符并规范化字符串
    cleaned_filename = re.sub(r"[^\w\-_.() ]", "", filename)
    cleaned_filename = (
        unicodedata.normalize("NFKD", cleaned_filename)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )
    cleaned_filename = cleaned_filename.replace(" ", "_")
    return cleaned_filename


@app.route("/kp-abstract/upload/train_mp4", methods=["POST"])
def upload_train_mp4():
    # 判断mp4视频文件是否为空
    if not os.path.exists(app.config["UPLOAD_TRAIN_VIDEO_FOLDER"]):
        os.makedirs(app.config["UPLOAD_TRAIN_VIDEO_FOLDER"])
    file = request.files["file"]
    fname = clean_filename(file.filename)
    if allowed_file(fname):
        # 保存视频文件
        video_path = os.path.join(app.config["UPLOAD_TRAIN_VIDEO_FOLDER"], fname)
        file.save(video_path)

        # 调用子进程运行帧图片提取脚本
        subprocess.run(
            ["python", "get_train_imgs.py", os.path.splitext(fname)[0]], check=True
        )
    else:
        # 保存视频文件
        video_path = os.path.join(
            app.config["UPLOAD_TRAIN_VIDEO_FOLDER"], "default_name"
        )
        file.save(video_path)

        # 调用子进程运行帧图片提取脚本
        subprocess.run(["python", "get_train_imgs.py", "default_name"], check=True)
    return "OK!"


# 展示视频训练图片的接口
@app.route("/kp-abstract/show_train_img")
def show_img():
    get_data = request.args.to_dict()
    folder_name = get_data.get("folder_name")

    # 构建文件夹的完整路径
    folder_path = os.path.join(app.config["DOWNLOAD_TRAIN_IMG_FILE"], folder_name)

    # 获取文件夹中的所有文件
    img_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # 返回文件列表给前端
    return jsonify({"img_files": img_files})


# 为了展示在前台页面封装好数据，加上key和value字段
@app.route("/kp-abstract/abstract/show")
def abstract_batch_show():
    get_data = request.args.to_dict()
    shape = get_data.get("shape")
    if shape == "mp4":
        video_name = get_data.get("video_name")
        program = "3_match_sj_zr.py"
        print(video_name)
        output = subprocess.check_output(
            ["python", program, video_name], encoding="gbk"
        )
        # 编写正则表达式模式来匹配需要的内容
        segments_match = re.search(r"key_list\s*=\s*\[(.*?)\]", output)
        res_match = re.search(r"res_list\s*=\s*\[(.*?)\)\]", output)
        # tp_match = re.search(r'tp_res\s*=\s*\[(.*?)\}\]', output)
        if segments_match:
            segments_res = eval(segments_match.group(1))
            segments = segments_res
        if res_match:
            res_list_text = eval("[" + res_match.group(1) + ")]")
        # if tp_match:
        #     res_tp_text = eval('[' + tp_match.group(1) + '}]')
        res = []
        i = 0
        for item in res_list_text:
            res.append((f"知识点 {i+1}:{item[0]}", item[1], segments[i]))
            i += 1
        return res
    return "error"


# 下载
@app.route("/kp-abstract/download/kp")
def download_kp():
    get_data = request.args.to_dict()
    shape = get_data.get("shape")
    if shape == "mp4":
        video_name = get_data.get("video_name")
        program = "3_match_sj.py"
        output = subprocess.check_output(
            ["python", program, video_name], encoding="gbk"
        )
        # 编写正则表达式模式来匹配需要的内容
        segments_match = re.search(r"key_list\s*=\s*\[(.*?)\]", output)
        res_match = re.search(r"res_list\s*=\s*\[(.*?)\)\]", output)
        # tp_match = re.search(r'tp_res\s*=\s*\[(.*?)\}\]', output)
        if segments_match:
            segments_res = eval(segments_match.group(1))
            segments = segments_res
        if res_match:
            res_list_text = eval("[" + res_match.group(1) + ")]")
        # if tp_match:
        #     res_tp_text = eval('[' + tp_match.group(1) + '}]')
        res = []
        i = 0
        for item in res_list_text:
            res.append((f"知识点 {i + 1}:{item[0]}", item[1], item[2], segments[i]))
            i += 1
        return res


# 下载五元组文件
@app.route("/kp-abstract/download/tuple")
def download_five_tuple():
    get_data = request.args.to_dict()
    shape = get_data.get("shape")
    if shape == "mp4":
        # 指定本地文本文件的路径
        # 使用Flask的send_file函数将文本文件发送给客户端
        return send_file(DOWNLOAD_KP_FILE, as_attachment=True)


# # 设置知识点个数
# @app.route('/kp-abstract/update/number')
# def set_out_number():
#     get_data = request.args.to_dict()
#     number = get_data.get('num')
#     res = kp_abstract.update_number(int(number))
#     # print('传过来的数字是', number)
#     if res == 1:
#         return "update success!"
#     else:
#         return "update defeat!"


# 给图谱调用的五元组格式数据接口
@app.route("/kp-abstract/graph/tuple")
def get_five_tuple():
    get_data = request.args.to_dict()
    shape = get_data.get("shape")
    if shape == "mp4":
        video_name = get_data.get("video_name")
        program = "3_match_sj.py"
        output = subprocess.check_output(
            ["python", program, video_name], encoding="gbk"
        )
        # 编写正则表达式模式来匹配需要的内容
        tp_match = re.search(r"tp_res\s*=\s*\[(.*?)\}\]", output)
        if tp_match:
            res_tp = eval("[" + tp_match.group(1) + "}]")
        data = res_tp
        json_data = json.dumps(data)
        url = "http://127.0.0.1:8080/generateGraph"
        header = {"Content-Type": "application/json"}
        r = requests.post(url, data=json_data, headers=header)
        return r.text
    return "error"


RES_DIR = "res"  # 假设视频文件在res目录下


@app.route("/kp-abstract/download/all_videos")
def download_video():
    video_filename = request.args.get("video_filename")
    return send_from_directory(directory=RES_DIR, path=video_filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("8010"), debug=True)
