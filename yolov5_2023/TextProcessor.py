import glob
import os
import re
import sys
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from nltk.tokenize import word_tokenize


class TextProcessor:
    def __init__(self, video_name):
        self.video_name = video_name
        self.image_dir = os.path.join("./yolo_res", video_name)
        self.label_dir = os.path.join("./yolo_res", video_name, "labels")
        self.time_interval = 1
        self.input_dir = os.path.join("./result_all_txt", video_name)
        self.output_dir = os.path.join("./label_hb_txt", video_name)
        self.folder_path = os.path.join("./result_all_txt", video_name)
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        self.label_files = [f for f in os.listdir(self.label_dir) if f.endswith(".txt")]
        self.nums_pic = len(self.label_files)
        self.no_labels_list = []
        self.number_list = []
        self.textL = 0

    def check_first_column(self,file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # 检查是否存在至少一个分隔符
            if not line.split():
                continue
            first_digit = int(line.split()[0])
            if first_digit in {0, 1, 2}:
                return False
        return True

    def find_files_without_digits(self,directory_path):
        files_without_digits = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                if self.check_first_column(file_path):
                    files_without_digits.append(filename)
                    self.number_list.append(int(os.path.splitext(filename)[0].split("_")[-1]))
                    # os.remove(file_path)
                    # os.remove(os.path.join(self.input_dir,filename))
        print(self.number_list) 
        return files_without_digits
    
    # def find_missing_files(self):
    #     for img_file in self.image_files:
    #         txt_file = img_file[:-4] + ".txt"
    #         if txt_file not in self.label_files:
    #             file_name = os.path.splitext(img_file)[0]
    #             file_name = file_name.split("_")[-1]
    #             if file_name.isdigit() and len(file_name) == 3:
    #                 self.number_list.append(int(file_name))
    #                 try:
    #                     print(img_file)
    #                     os.remove(os.path.join("results", self.video_name, img_file))
    #                     print(f"{img_file} 图片已被删除")
    #                     os.remove(os.path.join(self.folder_path, txt_file))
    #                     print(f"{txt_file} 文件已被删除")
    #                 except Exception as e:
    #                     continue
    #     if self.number_list:
    #         # 调用函数合并连续数字
    #         merged_ranges = self.merge_continuous_numbers()
    #         # 打印结果
    #         # print(merged_ranges)

    # def merge_continuous_numbers(self):
    #     ranges = []
    #     start = end = self.number_list[0]

    #     for num in self.number_list[1:]:
    #         if num == end + 1:
    #             end = num
    #         else:
    #             ranges.append([start, end])
    #             start = end = num

    #     ranges.append([start, end])
    #     return ranges

    def point_not_in_ranges(self, point, ranges):
        for r in ranges:
            if point >= r[0] and point <= r[1]:
                return False
        return True

    def read_text_files(self):
        texts = []
        txt_files = sorted(
            [file for file in os.listdir(self.folder_path) if file.endswith(".txt")]
        )
        self.textL = len(txt_files)
        for file in txt_files:
            print(file)
            file_path = os.path.join(self.folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                text = re.sub(
                    r"(\d)(?=[\u4e00-\u9fff])|([\u4e00-\u9fff])(?=\d)|([\u4e00-\u9fff])(?=[.,，。!?])|([.,，。!?])(?=[\u4e00-\u9fff])",
                    r"\1 \2 \3 \4 ",
                    text,
                )
                # 定义要使用的分隔符
                separators = ["的", "与", "和"]
                # 用于存储结果的列表
                result2 = []
                # 遍历每个字符串
                print(text)
                words2 = list(jieba.cut(text,cut_all=False)) 
                # print(words2)
                # + word_tokenize(text)
                # 过滤空格和换行符
                words2 = [word for word in words2 if word.strip() != '' and word != '\n']

                # + word_tokenize(text)
                # print(words2)
                for item in words2:
                    # 初始化当前子集
                    subset = []

                    # 遍历字符串中的每个字符
                    for char in item:
                        # 如果字符是分隔词，则添加当前子集到结果列表，并重置子集
                        if char in separators:
                            if subset:
                                result2.append("".join(subset))
                            subset = []
                        else:
                            # 如果字符不是分隔词，则将字符添加到当前子集中
                            subset.append(char)

                    # 添加最后一个子集到结果列表
                    if subset:
                        result2.append("".join(subset))

                result2 = set(result2)

                result2 = [item for item in result2 if len(item) >= 2]
                # print(result2)
                my_txt = " ".join(result2)
                # my_txt = " ".join(words2)
                # print(my_txt)
                texts.append(my_txt)
        # print(texts)
        return texts

    def calculate_similarity(self, texts):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return similarity_matrix

    def process_similarity_scores(self, similarity_scores):
        similarity_scores = [similarity_scores[i - 1][i] for i in range(1, self.textL)]
        print(similarity_scores)
        sorted_data = sorted(enumerate(similarity_scores), key=lambda x: x[1])
        lowest_10 = [item for item in sorted_data if item[1] < 0.3 and item[0] > 1]
        print("low10:")
        print(lowest_10)
        data = lowest_10
        data = lowest_10 = [item for item in data if item[0] not in self.number_list]
        new_data = []

        for item in data:
            add_item = True
            if item[0] < 1:
                add_item = False
            elif item[0] in [x[0] for x in new_data]:
                add_item = False
            else:
                for existing_item in new_data:
                    if abs(item[0] - existing_item[0]) <= 1:
                        add_item = False
                        break
            if add_item:
                new_data.append(item)

        return new_data

    def which_file(self, xb):
        files = os.listdir(self.folder_path)
        # 按照文件名中的数字部分从小到大排序
        sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
        if len(sorted_files) >= xb:
            res_file = sorted_files[xb]  # 第2个文件，索引从0开始
            # print(sorted_files[0])
            file_order = int(os.path.splitext(res_file)[0].split("_")[-1])
            return file_order - 1
        else:
            print(f"下标错误，目录中没有足够的文件")
            return -1

    def generate_output_list(self, new_data):
        # 遍历列表，并修改第一个元组的值
        new_data = sorted(new_data, key=lambda x: x[0])
        for index, (first_value, second_value) in enumerate(new_data):
            new_data[index] = (self.which_file(first_value), second_value)
        print(new_data)
        input_list = new_data
        maxnum = self.nums_pic
        output_list = [(1, input_list[0][0] + 1)]
        output_list.extend(
            [
                (input_list[i][0] + 1, input_list[i + 1][0] + 1)
                for i in range(len(input_list) - 1)
            ]
        )
        output_list.append((input_list[-1][0] + 1, maxnum))
        print(output_list)
        return output_list

    def clear_txt_files(self, directory_path):
        txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
        for file_path in txt_files:
            os.remove(file_path)

    def merge_txt_files(self, start_idx, end_idx):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        merged_content = []
        for idx in range(start_idx, end_idx + 1):
            if idx not in self.number_list:
                file_pattern = f"*_{idx:03d}.txt"
                matching_files = glob.glob(os.path.join(self.input_dir, file_pattern))
                for matching_file in matching_files:
                    with open(matching_file, "r", encoding="utf-8") as file:
                        content = file.read()
                        merged_content.append(content)

        output_file_name = f"{start_idx}_{end_idx}.txt"
        output_file_path = os.path.join(self.output_dir, output_file_name)

        with open(output_file_path, "w", encoding="utf8") as output_file:
            output_file.write("\n".join(merged_content))

        print(
            f"Merged content for knowledge point ({start_idx}, {end_idx}) written to {output_file_path}"
        )

    def process_knowledge_points(self):
        self.no_labels_list = self.find_files_without_digits(self.label_dir)
        knowledge_points = self.generate_output_list(
            self.process_similarity_scores(
                self.calculate_similarity(self.read_text_files())
            )
        )
        # print(knowledge_points)
        start_time = 0

        self.clear_txt_files(self.output_dir)

        strings = []
        for idx, (start_idx, end_idx) in enumerate(knowledge_points):
            knowledge_start_time = start_time + start_idx * self.time_interval
            knowledge_end_time = start_time + end_idx * self.time_interval
            if start_idx != 1:
                start_idx = start_idx + 1
            end_idx = end_idx
            self.merge_txt_files(start_idx, end_idx)
            strings.append(
                f"知识点 {idx + 1}: 开始帧图片索引：{start_idx}，结束帧图片索引：{end_idx}，"
                f"开始时间：{knowledge_start_time}秒，结束时间：{knowledge_end_time}秒."
            )
        print(strings)


if __name__ == "__main__":
    video_name_arg = sys.argv[1] if len(sys.argv) > 1 else "4-4"
    text_processor = TextProcessor(video_name_arg)
    text_processor.process_knowledge_points()
