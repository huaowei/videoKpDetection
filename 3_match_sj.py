# -*- coding: utf-8 -*-
import codecs
import json
import os
import re
import sys

import jieba
import pymysql
import read_db as read_db


# 02_02 5_3_sj 3.6 5.1
video_name = "3.6"
# video_name = sys.argv[1]
# print(video_name)
internal_time = 10

input_dir = os.path.join("label_hb_txt", video_name)
# print(knowledge_points)
directory_path = os.path.join("label_hb_txt", video_name)
directory_path = os.path.join("label_hb_txt", video_name)
# 实例化数据库对象sor
sor = read_db.DB(
    host="47.102.204.29",
    user="shuishandb",
    password="midStage2022",
    database="self_adaptive_learning",
    cursorclass=pymysql.cursors.DictCursor,
)
stopwords_filepath = "./stopwords.txt"

# 结果保存目录
data_filename = "video_yolo_result/labels/01_001.txt"
result_filename = "result/01_001_0.txt"
data_filename = "video_yolo_result/labels/01_001.txt"
result_filename = "result/01_001_0.txt"

flag_have_first_title = False
# 父节点
father = -1


def find_line_with_largest_number(filename):
    max_number = float("-inf")  # Initialize the maximum number as negative infinity
    max_number = float("-inf")  # Initialize the maximum number as negative infinity
    line_number = 0
    max_line_number = None

    with open(filename, "r") as file:
    with open(filename, "r") as file:
        for line in file:
            line_number += 1
            # Split the line by whitespace and get the last element as a number
            numbers = line.strip().split()
            if numbers:
                try:
                    first_number = int(numbers[0])
                    last_number = float(numbers[-1])
                    # print(first_number)
                    if first_number == 1:
                        continue  # Skip lines with first value equal to 1
                    # print(last_number)
                    if last_number > max_number:
                        max_number = last_number
                        max_line_number = line_number
                except ValueError:
                    pass  # Ignore lines with non-numeric last values

    return max_line_number


def get_line_from_file(filename, line_number):
    with open(filename, "r") as file:
    with open(filename, "r") as file:
        for i, line in enumerate(file, 1):
            if i == line_number:
                return line.strip()

    return None



def spilt_kp_word(stopwords_filepath):
    sql = "SELECT kp_id, kp_name FROM kp_knowledge_point WHERE type = 3 ;"
    sql = "SELECT kp_id, kp_name FROM kp_knowledge_point WHERE type = 3 ;"
    # Assuming `sor` is defined and used to fetch data from the database.
    res = sor.get_all(sql)
    stopwords = set(
        [line.strip() for line in codecs.open(stopwords_filepath, "r", "utf-8")]
    )
    stopwords = set(
        [line.strip() for line in codecs.open(stopwords_filepath, "r", "utf-8")]
    )
    # print(stopwords)

    filtered_res = []
    for word_dict in res:
        word = word_dict["kp_name"]
        word = word_dict["kp_name"]
        # 过滤掉'kp_name'是'数据'、'信息'和'分类'
        if word not in ["数据", "信息", "分类"]:
        if word not in ["数据", "信息", "分类"]:
            seg_list = jieba.cut(word)
            term_list = []
            for seg in seg_list:
                # 设置停用词 and filter out segments with length less than 2
                if seg not in stopwords and len(seg) >= 2:
                    term_list.append(seg)
            # # Add 'kp_name' itself to the term_list
            # term_list.append(word)
            # 使用set()函数将list转换为集合，集合自动去除重复项
            unique_list = list(set(term_list))
            word_dict["segment"] = unique_list
            word_dict["segment"] = unique_list
            filtered_res.append(word_dict)
    # print(filtered_res)

    return filtered_res


def spilt_kp_word_true(result_lists, stopwords_filepath):
    # Assuming `sor` is defined and used to fetch data from the database.
    res = result_lists
    stopwords = set(
        [line.strip() for line in codecs.open(stopwords_filepath, "r", "utf-8")]
    )
    stopwords = set(
        [line.strip() for line in codecs.open(stopwords_filepath, "r", "utf-8")]
    )
    # print(stopwords)

    filtered_res = []
    for word_dict in res:
        word = word_dict["kp_name"]
        word = word_dict["kp_name"]
        # 过滤掉'kp_name'是'数据'、'信息'和'分类'
        if word not in ["数据", "信息", "分类"]:
        if word not in ["数据", "信息", "分类"]:
            seg_list = jieba.cut(word)
            term_list = []
            for seg in seg_list:
                # 设置停用词 and filter out segments with length less than 2
                if seg not in stopwords and len(seg) >= 2:
                    term_list.append(seg)
            # # Add 'kp_name' itself to the term_list
            # term_list.append(word)
            # 使用set()函数将list转换为集合，集合自动去除重复项
            unique_list = list(set(term_list))
            word_dict["segment"] = unique_list
            word_dict["segment"] = unique_list
            filtered_res.append(word_dict)
    # print(filtered_res)

    return filtered_res


def count_kp_occurrences(text_file_path, knowledge_points):
    # print(text_file_path)
    # Read the content of the text file
    with open(text_file_path, "r", encoding="utf8") as file:
    with open(text_file_path, "r", encoding="utf8") as file:
        text_content = file.read()

    # Create a dictionary to store the occurrences of each knowledge point
    kp_occurrences = {kp["kp_name"]: (0, kp["kp_id"]) for kp in knowledge_points}
    kp_occurrences = {kp["kp_name"]: (0, kp["kp_id"]) for kp in knowledge_points}

    # Count occurrences in the text for each knowledge point and its segments
    for kp in knowledge_points:
        kp_id = kp["kp_id"]
        kp_name = kp["kp_name"]
        segment = kp["segment"]
        kp_id = kp["kp_id"]
        kp_name = kp["kp_name"]
        segment = kp["segment"]
        kp_occurrences[kp_name] = (31 * text_content.count(kp_name), kp_id)
        for word in segment:
            kp_occurrences[kp_name] = (
                kp_occurrences[kp_name][0] + text_content.count(word),
                kp_id,
            )
            kp_occurrences[kp_name] = (
                kp_occurrences[kp_name][0] + text_content.count(word),
                kp_id,
            )
        # print(kp_occurrences[kp_name][0])
        if kp_occurrences[kp_name][0] < 10:
            kp_occurrences[kp_name] = (0, kp_id)

    # Filter and return knowledge points with occurrences greater than 0
    kp_occurrences_filtered = {
        kp_name: count_id
        for kp_name, count_id in kp_occurrences.items()
        if count_id[0] > 0
    }
    kp_occurrences_filtered = {
        kp_name: count_id
        for kp_name, count_id in kp_occurrences.items()
        if count_id[0] > 0
    }

    # Sort knowledge points by occurrence count in descending order
    sorted_kp_occurrences = dict(
        sorted(
            kp_occurrences_filtered.items(), key=lambda item: item[1][0], reverse=True
        )
    )
    sorted_kp_occurrences = dict(
        sorted(
            kp_occurrences_filtered.items(), key=lambda item: item[1][0], reverse=True
        )
    )
    return sorted_kp_occurrences


def process_directory(directory_path, knowledge_points, n):
    results = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)

            # 调用count_kp_occurrences函数来处理当前文件
            kp_occurrences = count_kp_occurrences(file_path, knowledge_points)

            # 打印当前文件的前n个知识点及其出现次数
            top_n_knowledge_points = dict(list(kp_occurrences.items())[:n])
            file_results = {
                "filename": filename,
                "knowledge_points": {
                    kp_name: count for kp_name, count in top_n_knowledge_points.items()
                },
                "knowledge_points": {
                    kp_name: count for kp_name, count in top_n_knowledge_points.items()
                },
            }
            results[filename] = file_results
    # 按照filename中_前面的数值进行排序
    sorted_results = dict(
        sorted(results.items(), key=lambda item: int(item[0].split("_")[0]))
    )
    sorted_results = dict(
        sorted(results.items(), key=lambda item: int(item[0].split("_")[0]))
    )

    # 将结果转换成JSON对象并返回
    return json.dumps(sorted_results, indent=4, ensure_ascii=False)


def select_knowledge_points(json_data):
    selected_knowledge_points = []
    previous_knowledge_points = set()
    json_data = json.loads(json_data)

    for filename, data in json_data.items():
        knowledge_points = data["knowledge_points"]
        iterator = iter(knowledge_points.items())

        fp, fv = next(iterator)
        detail_fp = data["knowledge_points"][fp]
        try:
            sp, sv = next(iterator)
            detail_sp = data["knowledge_points"][sp]
        except StopIteration:
            sp, sv = None, None
            detail_sp = None

        if fp not in previous_knowledge_points:
            selected_knowledge_points.append((filename, fp, detail_fp))
            previous_knowledge_points.add(fp)
        elif sp and sp not in previous_knowledge_points:
            selected_knowledge_points.append((filename, sp, detail_sp))
            previous_knowledge_points.add(sp)

    return selected_knowledge_points



def select_knowledge_points_no(json_data):
    selected_knowledge_points = []
    json_data = json.loads(json_data)
    for filename, data in json_data.items():
        knowledge_points = data["knowledge_points"]
        iterator = iter(knowledge_points.items())
        fp, fv = next(iterator)

        try:
            sp, sv = next(iterator)
        except StopIteration:
            sp, sv = None, None

        selected_knowledge_points.append((filename, fp))

    return selected_knowledge_points




# 强制选择方式
flag_have_first_title = False
knowledge_points = spilt_kp_word(stopwords_filepath)




# 自定义比较函数，根据键值中的数字进行排序
def custom_sort(item):
    key = item[0]
    # 从键值中提取数字部分
    digits = [int(s) for s in re.findall(r"\d+", key)]
    digits = [int(s) for s in re.findall(r"\d+", key)]
    # 返回键值中的第一个数字
    return digits[0]




# 定义一个排序函数，根据每个知识点的第一个数字进行排序
def sort_key(item):
    return item[1][0]



def merge_data_xl_same(data):
    # 使用 eval() 将字符串转换为字典
    # try:
    #     data = eval(data)
    # except SyntaxError:
    #     raise ValueError("Invalid data string format, unable to convert to dictionary.")
    # 新的合并后的数据结构
    merged_data = {}
    current_key = None

    for key, value in data.items():
        if current_key is None:
            current_key = key
            merged_data[key] = value
        else:
            # 获取当前文本的知识点信息
            current_knowledge_points = value["knowledge_points"]
            # 获取前一个文本的知识点信息
            previous_knowledge_points = merged_data[current_key]["knowledge_points"]

            ckey, pkey = "", ""
            for key1, value1 in current_knowledge_points.items():
                ckey = key1
                break
            for key2, value2 in previous_knowledge_points.items():
                pkey = key2
                break

            # 检查是否具有相同的知识点
            if ckey == pkey:
                # 合并文件名
                current_filename_parts = current_key.split("_")
                key_parts = key.split("_")
                merged_filename = f"{current_filename_parts[0]}_{key_parts[1]}"
                merged_data[current_key]["filename"] = merged_filename

                # 合并知识点信息
                merged_knowledge_points = {}
                for k, v in current_knowledge_points.items():
                    if k in previous_knowledge_points:
                        merged_knowledge_points[k] = [
                            v[0] + previous_knowledge_points[k][0],
                            v[1],
                        ]
                        merged_knowledge_points[k] = [
                            v[0] + previous_knowledge_points[k][0],
                            v[1],
                        ]
                    else:
                        merged_knowledge_points[k] = v

                # 添加前一个文本中存在但当前文本中不存在的知识点
                for k, v in previous_knowledge_points.items():
                    if k not in current_knowledge_points:
                        merged_knowledge_points[k] = v

                merged_data[current_key]["knowledge_points"] = merged_knowledge_points
            else:
                # 如果知识点不同，创建新的记录
                current_key = key
                merged_data[key] = value
    for filename, file_data in merged_data.items():
        # 检查文件数据是否包含knowledge_points
        if "knowledge_points" in file_data:
        if "knowledge_points" in file_data:
            # 对knowledge_points字典按照sort_key进行排序
            sorted_knowledge_points = dict(
                (k, v)
                for k, v in sorted(
                    file_data["knowledge_points"].items(),
                    file_data["knowledge_points"].items(),
                    key=sort_key,
                    reverse=True,  # 降序排序
                    reverse=True,  # 降序排序
                )
            )

            # 更新原始数据中的knowledge_points
            file_data["knowledge_points"] = sorted_knowledge_points
            file_data["knowledge_points"] = sorted_knowledge_points
    return merged_data



def sort_data_by_filename(data):
    # print(data)
    # 使用 eval() 将字符串转换为字典
    # try:
    #     data = eval(data_str)
    # except SyntaxError:
    #     raise ValueError("Invalid data string format, unable to convert to dictionary.")

    # 存储已选择的知识点
    selected_knowledge_points = set()

    # 存储已选择的文件名
    selected_filenames = set()

    # 提取知识点次数并按次数降序排列
    result = []
    while True:
        max_knowledge_point = None
        max_filename = None
        # print(data.items())
        for filename, knowledge_points in data.items():
            # 如果文件名已经被选择，则跳过
            # print(filename)
            if filename in selected_filenames:
                continue

            # 删除已选择的知识点
            for selected_kp in selected_knowledge_points:
                if selected_kp in knowledge_points["knowledge_points"]:
                    del knowledge_points["knowledge_points"][selected_kp]

            # 找到当前文件中次数最多的知识点
            if knowledge_points["knowledge_points"]:
                current_max = max(
                    knowledge_points["knowledge_points"].items(), key=lambda x: x[1][0]
                )
                if (
                    max_knowledge_point is None
                    or current_max[1][0] > max_knowledge_point[1][0]
                ):
                if (
                    max_knowledge_point is None
                    or current_max[1][0] > max_knowledge_point[1][0]
                ):
                    max_knowledge_point = current_max
                    max_filename = filename

        if max_knowledge_point:
            # 将选择的知识点添加到已选择的知识点集合中
            selected_knowledge_points.add(max_knowledge_point[0])

            # 将选择的文件名添加到已选择的文件名集合中
            selected_filenames.add(max_filename)

            # 格式化结果并添加到结果列表中
            result.append(
                (max_filename, max_knowledge_point[0], max_knowledge_point[1])
            )
            result.append(
                (max_filename, max_knowledge_point[0], max_knowledge_point[1])
            )

        else:
            break
    # 按filename排序
    result.sort(key=lambda x: x[0])
    # 使用sorted函数进行排序，key参数使用自定义比较函数
    sorted_data = sorted(result, key=custom_sort)

    return sorted_data


n = 3
data_ = process_directory(directory_path, knowledge_points, n)
print(data_)
# 添加筛选机制
result = list(
    [
        value[1]
        for inner_dict in eval(data_).values()
        for value in inner_dict["knowledge_points"].values()
    ]
)
result = list(
    [
        value[1]
        for inner_dict in eval(data_).values()
        for value in inner_dict["knowledge_points"].values()
    ]
)
# print(result)
query_test = "SELECT parent_id FROM kp_knowledge_point WHERE kp_id = %s;"
count_kp_parent = {}
for item in result:
    parent_id_info = sor.get_one_2(query_test, (item,))
    # print(parent_id_info)
    # 提取 parent_id 的值
    parent_id = parent_id_info.get("parent_id")
    parent_id = parent_id_info.get("parent_id")

    if parent_id is not None:
        if parent_id in count_kp_parent:
            count_kp_parent[parent_id] += 1
        else:
            count_kp_parent[parent_id] = 1

# 打印统计结果
# for parent_id, count in count_kp_parent.items():
#     print(f"Parent ID {parent_id}: {count} times")
# 找到出现次数最多的 parent_id
max_parent_id = max(count_kp_parent, key=count_kp_parent.get)
max_count = count_kp_parent[max_parent_id]

print(f"Parent ID {max_parent_id} has the highest count: {max_count} times")
root_parent_id = sor.get_one_2(query_test, (max_parent_id,)).get("parent_id")
root_parent_id = sor.get_one_2(query_test, (max_parent_id,)).get("parent_id")
# print(root_parent_id)
query_l2 = "SELECT kp_id FROM kp_knowledge_point WHERE parent_id = %s;"
l2_id = sor.get_all_2(query_l2, (root_parent_id,))
# print(l2_id)
# 使用 sorted 函数按照 'kp_id' 的值排序
l2_id = sorted(l2_id, key=lambda x: x["kp_id"])
l2_id = sorted(l2_id, key=lambda x: x["kp_id"])
# 存放有效的子节点序号
effective_list = []
for item in l2_id:
    kp_id = item["kp_id"]
    kp_id = item["kp_id"]
    effective_list.append(sor.get_all_2(query_l2, (kp_id,)))
    # print(f'kp_id: {kp_id}')
# print(effective_list)
# 创建新的数据结构用于存储筛选后的元素
filtered_data = {}

# 遍历数据结构并筛选出内层数据出现在指定的列表中的元素
for key, inner_dict in eval(data_).items():
    filtered_inner_dict = {}
    for inner_key, value in inner_dict["knowledge_points"].items():
        if value[1] in [
            item["kp_id"] for sublist in effective_list for item in sublist
        ]:
    for inner_key, value in inner_dict["knowledge_points"].items():
        if value[1] in [
            item["kp_id"] for sublist in effective_list for item in sublist
        ]:
            filtered_inner_dict[inner_key] = value
    if filtered_inner_dict:
        filtered_data[key] = {
            "filename": inner_dict["filename"],
            "knowledge_points": filtered_inner_dict,
            "knowledge_points": filtered_inner_dict,
        }

# 打印筛选后的数据结构
# print(filtered_data)

# 合并处理
sorted_data = sort_data_by_filename(merge_data_xl_same(filtered_data))
# print(sorted_data)
# 遍历数据
begin = 1
for i in range(1, len(sorted_data)):
    prev_key = sorted_data[i - 1][0]  # 前一个key
    prev_seq = int(prev_key.split("_")[1].split(".")[0])  # 前一个key的序号
    prev_key = sorted_data[i - 1][0]  # 前一个key
    prev_seq = int(prev_key.split("_")[1].split(".")[0])  # 前一个key的序号

    current_key = sorted_data[i][0]  # 当前key
    current_seq = int(current_key.split("_")[0])  # 当前key的序号
    current_seq = int(current_key.split("_")[0])  # 当前key的序号
    # 检查是否连续
    if current_seq != prev_seq + 1:
        # 更新当前key的序号，使其连续
        current_key = f"{begin}_{current_seq-1}.txt"
        sorted_data[i - 1] = (current_key, sorted_data[i - 1][1], sorted_data[i - 1][2])
        current_key = f"{begin}_{current_seq-1}.txt"
        sorted_data[i - 1] = (current_key, sorted_data[i - 1][1], sorted_data[i - 1][2])
    begin = current_seq
# 打印更新后的数据
print("res_list= {}".format(sorted_data))

key_list = [
    (
        int(key.split("_")[0]) * internal_time,
        int(key.split("_")[1].split(".")[0]) * internal_time,
    )
    for key, _, _ in sorted_data
]
key_list = [
    (
        int(key.split("_")[0]) * internal_time,
        int(key.split("_")[1].split(".")[0]) * internal_time,
    )
    for key, _, _ in sorted_data
]
print("key_list= {}".format(key_list))
selected_knowledge_points = sorted_data
# selected_knowledge_points = select_knowledge_points(process_directory(directory_path, knowledge_points, n))
# selected_knowledge_points = select_knowledge_points_no(process_directory(directory_path, knowledge_points, n))
# print(selected_knowledge_points)


# 需要将输出的知识点内容转换成五元组内容
def dict_convert_tuple(selected_knowledge_points):
    # 定义一个返回的空列表
    tuple_kp_dicts_list = []
    # 遍历返回的具体知识点，课件名信息没用了，遍历value
    # 合并多个列表
    kp_dicts_list = selected_knowledge_points
    # 存储字典的列表
    elem_to_object_dict_list = []
    object_to_main_dict_list = []
    main_to_exact_dict_list = []
    for item in kp_dicts_list:
        query_1 = (
            "SELECT kp_id FROM kp_knowledge_point WHERE kp_name = %s AND type = 3;"
        )
        query_1 = (
            "SELECT kp_id FROM kp_knowledge_point WHERE kp_name = %s AND type = 3;"
        )
        kp_name_to_search = item[1]
        kp_id = sor.get_one_2(query_1, (kp_name_to_search,))
        # print(kp_id)

        # 通过kp_id查主题知识点和知识对象
        query_2 = "SELECT kp_name FROM kp_knowledge_point kp, ( SELECT @id as id, (SELECT @id := parent_id FROM kp_knowledge_point WHERE kp_id = @id) as pid, @l := @l + 1 as level FROM kp_knowledge_point, (SELECT @id := %s, @l := 0) b WHERE @id > 1 ) a WHERE kp.kp_id = a.pid AND kp.kp_name is not NULL;"
        query_2_params = kp_id["kp_id"]
        query_2_params = kp_id["kp_id"]
        kp_name_list = sor.get_all_2(query_2, (query_2_params,))
        # print(kp_name_list)

        # 通过kp_id查知识单元
        query_0 = "SELECT ke.knowledge_element_name FROM kp_knowledge_point kp, kp_ke_relationship kkr, kp_knowledge_element ke WHERE kp.kp_id = kkr.chlid_id AND kkr.parent_id = ke.knowledge_element_id AND kp_id = ( SELECT kp.kp_id FROM( SELECT @id as id, (SELECT @id := parent_id FROM kp_knowledge_point WHERE kp_id = @id) as pid, @l := @l + 1 as level FROM kp_knowledge_point, (SELECT @id := %s, @l := 0) b WHERE @id > 0  LIMIT 3 ) kp_tmp, kp_knowledge_point kp WHERE kp_tmp.id = kp.kp_id AND type = 1 ORDER BY level );"
        query_0_params = kp_id["kp_id"]
        query_0_params = kp_id["kp_id"]
        ke_name = sor.get_one_2(query_0, (query_0_params,))
        # print(ke_name)
        # 先返回知识单元和知识对象的五元组
        # elem_to_object_dict = {}
        # elem_to_object_dict['type1'] = "知识单元"
        # elem_to_object_dict['element1'] = ke_name['knowledge_element_name']
        # elem_to_object_dict['type2'] = "知识对象"
        # elem_to_object_dict['element2'] = kp_name_list[1]['kp_name']
        # elem_to_object_dict['relation'] = "包含"
        # elem_to_object_dict_list.append(elem_to_object_dict)
        # 然后返回知识对象和主题知识点的五元组
        object_to_main_dict = {}
        object_to_main_dict["type1"] = "知识对象"
        object_to_main_dict["element1"] = kp_name_list[1]["kp_name"]
        object_to_main_dict["type2"] = "主题知识点"
        object_to_main_dict["element2"] = kp_name_list[0]["kp_name"]
        object_to_main_dict["relation"] = "包含"
        object_to_main_dict["type1"] = "知识对象"
        object_to_main_dict["element1"] = kp_name_list[1]["kp_name"]
        object_to_main_dict["type2"] = "主题知识点"
        object_to_main_dict["element2"] = kp_name_list[0]["kp_name"]
        object_to_main_dict["relation"] = "包含"
        object_to_main_dict_list.append(object_to_main_dict)
        # 最后返回主题知识点和具体知识点的五元组
        main_to_exact_dict = {}
        main_to_exact_dict["type1"] = "主题知识点"
        main_to_exact_dict["element1"] = kp_name_list[0]["kp_name"]
        main_to_exact_dict["type2"] = "具体知识点"
        main_to_exact_dict["element2"] = item[1]
        main_to_exact_dict["relation"] = "包含"
        main_to_exact_dict["type1"] = "主题知识点"
        main_to_exact_dict["element1"] = kp_name_list[0]["kp_name"]
        main_to_exact_dict["type2"] = "具体知识点"
        main_to_exact_dict["element2"] = item[1]
        main_to_exact_dict["relation"] = "包含"
        main_to_exact_dict_list.append(main_to_exact_dict)
    # 分别对elem_to_object_dict_list, object_to_main_dict_list, main_to_exact_dict_list去重
    unrepeated_elem_to_object_dict_list = []
    unrepeated_object_to_main_dict_list = []
    unrepeated_main_to_exact_dict_list = []
    for i in elem_to_object_dict_list:
        if i not in unrepeated_elem_to_object_dict_list:
            unrepeated_elem_to_object_dict_list.append(i)
    for j in object_to_main_dict_list:
        if j not in unrepeated_object_to_main_dict_list:
            unrepeated_object_to_main_dict_list.append(j)
    for k in main_to_exact_dict:
        if k not in unrepeated_main_to_exact_dict_list:
            unrepeated_main_to_exact_dict_list.append(k)
    # 合并三个列表
    tuple_kp_dicts_list = (
        unrepeated_elem_to_object_dict_list
        + unrepeated_object_to_main_dict_list
        + main_to_exact_dict_list
    )
    tuple_kp_dicts_list = (
        unrepeated_elem_to_object_dict_list
        + unrepeated_object_to_main_dict_list
        + main_to_exact_dict_list
    )
    return tuple_kp_dicts_list


def dict_convert_tuple_three(selected_knowledge_points):
    # 定义一个返回的空列表
    tuple_kp_dicts_list = []
    # 遍历返回的具体知识点，课件名信息没用了，遍历value
    # 合并多个列表
    kp_dicts_list = selected_knowledge_points
    # 存储字典的列表
    elem_to_object_dict_list = []
    object_to_main_dict_list = []
    main_to_exact_dict_list = []
    for item in kp_dicts_list:
        # query_1 = "SELECT kp_id FROM kp_knowledge_point WHERE kp_name = %s AND type = 3;"
        # kp_name_to_search = item[1]
        # kp_id = sor.get_one_2(query_1, (kp_name_to_search,))

        kp_id = item[2][1]

        # 通过kp_id查主题知识点和知识对象
        query_2 = "SELECT kp_name FROM kp_knowledge_point kp, ( SELECT @id as id, (SELECT @id := parent_id FROM kp_knowledge_point WHERE kp_id = @id) as pid, @l := @l + 1 as level FROM kp_knowledge_point, (SELECT @id := %s, @l := 0) b WHERE @id > 1 ) a WHERE kp.kp_id = a.pid AND kp.kp_name is not NULL;"
        query_2_params = kp_id
        kp_name_list = sor.get_all_2(query_2, (query_2_params,))
        # print(kp_name_list)

        # 通过kp_id查知识单元
        query_0 = "SELECT ke.knowledge_element_name FROM kp_knowledge_point kp, kp_ke_relationship kkr, kp_knowledge_element ke WHERE kp.kp_id = kkr.chlid_id AND kkr.parent_id = ke.knowledge_element_id AND kp_id = ( SELECT kp.kp_id FROM( SELECT @id as id, (SELECT @id := parent_id FROM kp_knowledge_point WHERE kp_id = @id) as pid, @l := @l + 1 as level FROM kp_knowledge_point, (SELECT @id := %s, @l := 0) b WHERE @id > 0  LIMIT 3 ) kp_tmp, kp_knowledge_point kp WHERE kp_tmp.id = kp.kp_id AND type = 1 ORDER BY level );"
        query_0_params = kp_id
        ke_name = sor.get_one_2(query_0, (query_0_params,))
        # print(ke_name)
        # 先返回知识单元和知识对象的五元组
        elem_to_object_dict = {}
        # elem_to_object_dict['element1'] = ke_name['knowledge_element_name']
        # elem_to_object_dict['element2'] = kp_name_list[1]['kp_name']
        # elem_to_object_dict['relation'] = "包含"
        # elem_to_object_dict_list.append(elem_to_object_dict)
        # 然后返回知识对象和主题知识点的五元组
        object_to_main_dict = {}
        object_to_main_dict["element1"] = kp_name_list[1]["kp_name"]
        object_to_main_dict["element2"] = kp_name_list[0]["kp_name"]
        object_to_main_dict["relation"] = "包含"
        object_to_main_dict["element1"] = kp_name_list[1]["kp_name"]
        object_to_main_dict["element2"] = kp_name_list[0]["kp_name"]
        object_to_main_dict["relation"] = "包含"
        object_to_main_dict_list.append(object_to_main_dict)
        # 最后返回主题知识点和具体知识点的五元组
        main_to_exact_dict = {}
        main_to_exact_dict["element1"] = kp_name_list[0]["kp_name"]
        main_to_exact_dict["element2"] = item[1]
        main_to_exact_dict["relation"] = "包含"
        main_to_exact_dict["element1"] = kp_name_list[0]["kp_name"]
        main_to_exact_dict["element2"] = item[1]
        main_to_exact_dict["relation"] = "包含"
        main_to_exact_dict_list.append(main_to_exact_dict)
    # 分别对elem_to_object_dict_list, object_to_main_dict_list, main_to_exact_dict_list去重
    unrepeated_elem_to_object_dict_list = []
    unrepeated_object_to_main_dict_list = []
    unrepeated_main_to_exact_dict_list = []
    for i in elem_to_object_dict_list:
        if i not in unrepeated_elem_to_object_dict_list:
            unrepeated_elem_to_object_dict_list.append(i)
    for j in object_to_main_dict_list:
        if j not in unrepeated_object_to_main_dict_list:
            unrepeated_object_to_main_dict_list.append(j)
    for k in main_to_exact_dict:
        if k not in unrepeated_main_to_exact_dict_list:
            unrepeated_main_to_exact_dict_list.append(k)
    # 合并三个列表
    tuple_kp_dicts_list = (
        unrepeated_elem_to_object_dict_list
        + unrepeated_object_to_main_dict_list
        + main_to_exact_dict_list
    )
    tuple_kp_dicts_list = (
        unrepeated_elem_to_object_dict_list
        + unrepeated_object_to_main_dict_list
        + main_to_exact_dict_list
    )
    return tuple_kp_dicts_list


# 保存为csv文件
def save_csv(file_name, tuple_kp_dict_list):
    try:
        with open(file_name, "w", encoding="utf-8") as file:
        with open(file_name, "w", encoding="utf-8") as file:
            for tuple_kp_dict in tuple_kp_dict_list:
                file.write(
                    tuple_kp_dict["type1"]
                    + ","
                    + tuple_kp_dict["element1"]
                    + ","
                    + tuple_kp_dict["type2"]
                    + ","
                    + tuple_kp_dict["element2"]
                    + ","
                    + tuple_kp_dict["relation"]
                    + "\n"
                )
                file.write(
                    tuple_kp_dict["type1"]
                    + ","
                    + tuple_kp_dict["element1"]
                    + ","
                    + tuple_kp_dict["type2"]
                    + ","
                    + tuple_kp_dict["element2"]
                    + ","
                    + tuple_kp_dict["relation"]
                    + "\n"
                )
            file.close()
        return 0
    except Exception as e:
        print(str(e))
        return -1


# 保存为csv文件
def save_csv_three(file_name, tuple_kp_dict_list):
    try:
        with open(file_name, "w", encoding="utf-8") as file:
        with open(file_name, "w", encoding="utf-8") as file:
            for tuple_kp_dict in tuple_kp_dict_list:
                file.write(
                    tuple_kp_dict["element1"]
                    + ","
                    + tuple_kp_dict["element2"]
                    + ","
                    + tuple_kp_dict["relation"]
                    + "\n"
                )
                    tuple_kp_dict["element1"]
                    + ","
                    + tuple_kp_dict["element2"]
                    + ","
                    + tuple_kp_dict["relation"]
                    + "\n"
                )
            file.close()
        return 0
    except Exception as e:
        print(str(e))
        return -1


# res = dict_convert_tuple_three(selected_knowledge_points)
res = dict_convert_tuple(selected_knowledge_points)
print("tp_res= {}".format(res))
save_csv_three("./kp_3.txt", res)
save_csv("./kp_5.txt", res)

save_csv_three("./kp_3.txt", res)
save_csv("./kp_5.txt", res)
