# import requests
#
# url = "http://localhost:8010/process"
# params = {'shape': 'mp4','video_name': '3.6'}
# # data = {"video_name": "3.6"}
#
# response = requests.post(url, params=params)
# print(response.json())
import requests

# 设置请求的URL和参数
url = 'http://localhost:8010/kp-abstract/abstract/show'  # 请根据你的实际Flask应用的URL进行调整
params = {'shape': 'mp4','video_name': '3.6'}
data = {'video_name': '3.6'}  # 请替换为实际的视频名

# 发送GET请求
response = requests.get(url, params=params, json=data)

# 检查响应状态码
if response.status_code == 200:
    # 如果响应状态码为200，表示成功
    result = response.json()  # 解析JSON响应
    print(result)
else:
    # 如果响应状态码不是200，表示出现了错误
    print(f"请求失败，状态码：{response.status_code}")

# 设置请求的URL和参数
# url = 'http://localhost:8010/kp-abstract/graph/tuple'  # 请根据你的实际Flask应用的URL进行调整
# params = {'shape': 'mp4'}
# data = {'video_name': '3.6'}  # 请替换为实际的视频名
#
# # 发送GET请求
# response = requests.get(url, params=params, json=data)
#
# # 检查响应状态码
# if response.status_code == 200:
#     # 如果响应状态码为200，表示成功
#     result = response.json()  # 解析JSON响应
#     print(result)
# else:
#     # 如果响应状态码不是200，表示出现了错误
#     print(f"请求失败，状态码：{response.status_code}")