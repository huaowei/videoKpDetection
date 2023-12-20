# -*- coding: utf-8 -*-
import os
import json
import re
import codecs
import pymysql
import jieba
from read_db import DB  # Assuming read_db is a module with a class named DB

class KnowledgePointProcessor:
    def __init__(self, video_name, internal_time=10):
        self.video_name = video_name
        self.internal_time = internal_time
        self.input_dir = os.path.join("label_hb_txt", video_name)
        self.directory_path = os.path.join('label_hb_txt', video_name)
        self.stopwords_filepath = "v0.0/stopwords.txt"
        self.result_filename = f'result/{video_name}_0.txt'
        self.flag_have_first_title = False
        self.father = -1
        self.knowledge_points = self.split_kp_word(self.stopwords_filepath)
        self.sor = DB(host='47.102.204.29', user='shuishandb', password='midStage2022',
                      database='self_adaptive_learning', cursorclass=pymysql.cursors.DictCursor)

    def find_line_with_largest_number(self, filename):
        max_number = float('-inf')  # Initialize the maximum number as negative infinity
        line_number = 0
        max_line_number = None

        with open(filename, 'r') as file:
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

    def get_line_from_file(self, filename, line_number):
        with open(filename, 'r') as file:
            for i, line in enumerate(file, 1):
                if i == line_number:
                    return line.strip()

        return None

    def split_kp_word(self, stopwords_filepath):
        sql = 'SELECT kp_id, kp_name FROM kp_knowledge_point WHERE type = 3 ;'
        # Assuming `sor` is defined and used to fetch data from the database.
        res = self.sor.get_all(sql)
        stopwords = set([line.strip() for line in codecs.open(stopwords_filepath, 'r', 'utf-8')])
        # print(stopwords)

        filtered_res = []
        for word_dict in res:
            word = word_dict['kp_name']
            # 过滤掉'kp_name'是'数据'、'信息'和'分类'
            if word not in ['数据', '信息', '分类']:
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
                word_dict['segment'] = unique_list
                filtered_res.append(word_dict)
        # print(filtered_res)

        return filtered_res

    def count_kp_occurrences(self, text_file_path, knowledge_points):
        with open(text_file_path, 'r', encoding='utf8') as file:
            text_content = file.read()

        # Create a dictionary to store the occurrences of each knowledge point
        kp_occurrences = {kp['kp_name']: (0, kp['kp_id']) for kp in knowledge_points}

        # Count occurrences in the text for each knowledge point and its segments
        for kp in knowledge_points:
            kp_id = kp['kp_id']
            kp_name = kp['kp_name']
            segment = kp['segment']
            kp_occurrences[kp_name] = (31 * text_content.count(kp_name), kp_id)
            for word in segment:
                kp_occurrences[kp_name] = (kp_occurrences[kp_name][0] + text_content.count(word), kp_id)
            # print(kp_occurrences[kp_name][0])
            if kp_occurrences[kp_name][0] < 10:
                kp_occurrences[kp_name] = (0, kp_id)

        # Filter and return knowledge points with occurrences greater than 0
        kp_occurrences_filtered = {kp_name: count_id for kp_name, count_id in kp_occurrences.items() if count_id[0] > 0}

        # Sort knowledge points by occurrence count in descending order
        sorted_kp_occurrences = dict(sorted(kp_occurrences_filtered.items(), key=lambda item: item[1][0], reverse=True))
        return sorted_kp_occurrences

    def process_directory(self, n):
        results = {}
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.directory_path, filename)

                # 调用count_kp_occurrences函数来处理当前文件
                kp_occurrences = self.count_kp_occurrences(file_path, self.knowledge_points)

                # 打印当前文件的前n个知识点及其出现次数
                top_n_knowledge_points = dict(list(kp_occurrences.items())[:n])
                file_results = {
                    "filename": filename,
                    "knowledge_points": {kp_name: count for kp_name, count in top_n_knowledge_points.items()}
                }
                results[filename] = file_results
        # 按照filename中_前面的数值进行排序
        sorted_results = dict(sorted(results.items(), key=lambda item: int(item[0].split('_')[0])))

        # 将结果转换成JSON对象并返回
        return json.dumps(sorted_results, indent=4, ensure_ascii=False)

    def sort_key(self, item):
        return item[1][0]

    def select_knowledge_points(self, json_data):
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

    def merge_data_xl_same(self, data):
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
                            merged_knowledge_points[k] = [v[0] + previous_knowledge_points[k][0], v[1]]
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
            if 'knowledge_points' in file_data:
                # 对knowledge_points字典按照sort_key进行排序
                sorted_knowledge_points = dict(
                    (k, v)
                    for k, v in sorted(
                        file_data['knowledge_points'].items(),
                        key=self.sort_key,
                        reverse=True  # 降序排序
                    )
                )

                # 更新原始数据中的knowledge_points
                file_data['knowledge_points'] = sorted_knowledge_points
        return merged_data

    def custom_sort(self, item):
        key = item[0]
        # 从键值中提取数字部分
        digits = [int(s) for s in re.findall(r'\d+', key)]
        # 返回键值中的第一个数字
        return digits[0]

    def sort_data_by_filename(self, data):
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
                    if max_knowledge_point is None or current_max[1][0] > max_knowledge_point[1][0]:
                        max_knowledge_point = current_max
                        max_filename = filename

            if max_knowledge_point:
                # 将选择的知识点添加到已选择的知识点集合中
                selected_knowledge_points.add(max_knowledge_point[0])

                # 将选择的文件名添加到已选择的文件名集合中
                selected_filenames.add(max_filename)

                # 格式化结果并添加到结果列表中
                result.append((max_filename, max_knowledge_point[0], max_knowledge_point[1]))

            else:
                break
        # 按filename排序
        result.sort(key=lambda x: x[0])
        # 使用sorted函数进行排序，key参数使用自定义比较函数
        sorted_data = sorted(result, key=self.custom_sort)

        return sorted_data

    def process_knowledge_points(self):
        n = 3
        data_ = self.process_directory(n)
        selected_knowledge_points = self.select_knowledge_points(data_)
        return self.merge_data_xl_same(selected_knowledge_points)

    # def get_knowledge_point_tuples(self, selected_knowledge_points):
    #     # Assuming you want to convert knowledge points to tuples
    #     # Implement your conversion logic here
    #     # Return the resulting list of tuples

if __name__ == "__main__":
    processor = KnowledgePointProcessor(video_name="02_01")
    knowledge_point_data = processor.process_knowledge_points()
    knowledge_point_tuples = processor.get_knowledge_point_tuples(knowledge_point_data)
    print(knowledge_point_tuples)
