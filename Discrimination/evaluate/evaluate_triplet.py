import json

# # 读取 JSON 文件
# with open('/home/ubuntu/meddataset/cholec_process/output4_triplets.json', 'r') as file:
#     json_data = json.load(file)

# # 统计不同三元组的函数
# def count_unique_triplets(json_data):
#     triplet_set = set()
    
#     for entry in json_data:
#         triplets_str = entry["triplets"]
#         if "triplet:" in triplets_str:
#             # 将字符串中的所有三元组提取出来并去除多余的部分
#             triplets = triplets_str.replace("triplet:", "").strip().split(", ")
#             for triplet in triplets:
#                 if triplet != "(no triplet)":  # 排除没有三元组的情况
#                     triplet_set.add(triplet)
    
#     return len(triplet_set), triplet_set

# # 计算并打印不同三元组的数量
# unique_triplet_count, unique_triplets = count_unique_triplets(json_data)
# unique_triplets_list = list(unique_triplets)
# print(unique_triplets_list)
# 假设您的 JSON 文件路径为 "translations_8.json"

with open('/home/ubuntu/meddataset/cholec/unistra_file/CholecT50-challenge-train/dict/triplet.txt', 'r') as file:
    lines = file.readlines()

def count_unique_triplets(lines):
    triplet_set = set()
    
    for line in lines:
        # 从每行中去掉前面的数字和冒号，保留三元组部分
        triplet_str = line.strip().split(':', 1)[-1]
        triplets = triplet_str.split(',')  # 将每个三元组拆分为单个元素
        # 将三元组格式化成 (x, y, z) 的样式
        formatted_triplet = f"({triplets[0]},{triplets[1]},{triplets[2]})"

        triplet_set.add(formatted_triplet)
    
    return len(triplet_set), triplet_set

# 计算并打印不同三元组的数量
unique_triplet_count, unique_triplets = count_unique_triplets(lines)
unique_triplets_list = list(unique_triplets)
print(len(unique_triplets_list))

file_path = "/home/ubuntu/meddataset/choelc_generate/combine/v2/triplet/inference_kmeans500v2/translations_final.json"

# 初始化两个二维数组，用于存储 generated 和 reference 中的多个三元组
generated_triplets = []
reference_triplets = []

# 读取 JSON 文件
with open(file_path, 'r') as file:
    data = json.load(file)

    # 遍历 translations 列表
    for item in data['translations']:
        # 提取 generated 中的多个三元组，并去掉 "triplet: " 前缀
        generated_str = item['generated'].replace("triplet: ", "").strip()

        # 将 generated 的三元组按逗号分割，并存储为列表
        generated_list = [triplet.strip() for triplet in generated_str.split("),") if triplet]
        generated_list = [triplet + ")" if not triplet.endswith(")") else triplet for triplet in generated_list]  # 确保每个三元组以 ")" 结尾
        generated_triplets.append(generated_list)

        # 提取 reference 中的多个三元组，并去掉 "triplet: " 前缀
        reference_str = item['reference']['triplets'].replace("triplet: ", "").strip()

        # 将 reference 的三元组按逗号分割，并存储为列表
        reference_list = [triplet.strip() for triplet in reference_str.split("),") if triplet]
        reference_list = [triplet + ")" if not triplet.endswith(")") else triplet for triplet in reference_list]  # 确保每个三元组以 ")" 结尾
        reference_triplets.append(reference_list)

# 输出结果，检查是否正确
# print("Generated Triplets:", generated_triplets)
# print("Reference Triplets:", reference_triplets)

from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, f1_score, precision_score, recall_score

# 假设你已经定义了 unique_triplets_list 和 y_true_triplets_list、y_pred_triplets_list
# unique_triplets_list 包含了所有可能的三元组

# 创建三元组到索引的映射
triplet_to_index = {triplet: idx for idx, triplet in enumerate(unique_triplets_list)}

# 示例多个样本的真实标签和预测标签
# 多个样本的真实值和预测值可以不同三元组
y_true_triplets_list = reference_triplets
y_pred_triplets_list = generated_triplets

# 初始化所有样本的多标签结果
y_true_matrix = []
y_pred_matrix = []

# 对每个样本的真实和预测标签进行编码
for y_true_triplets in y_true_triplets_list:
    y_true = [0] * len(unique_triplets_list)  # 初始化零向量
    if "no triplet" not in y_true_triplets:
        for triplet in y_true_triplets:
            if triplet in triplet_to_index:
                y_true[triplet_to_index[triplet]] = 1
    y_true_matrix.append(y_true)

for y_pred_triplets in y_pred_triplets_list:
    y_pred = [0] * len(unique_triplets_list)  # 初始化零向量
    if "no triplet" not in y_pred_triplets:
        for triplet in y_pred_triplets:
            if triplet in triplet_to_index:
                y_pred[triplet_to_index[triplet]] = 1
    y_pred_matrix.append(y_pred)

# 计算评估指标
subset_acc = accuracy_score(y_true_matrix, y_pred_matrix)
hamming = hamming_loss(y_true_matrix, y_pred_matrix)
jaccard = jaccard_score(y_true_matrix, y_pred_matrix, average='samples',zero_division=1)
f1 = f1_score(y_true_matrix, y_pred_matrix, average='micro')
precision = precision_score(y_true_matrix, y_pred_matrix, average='micro')
recall = recall_score(y_true_matrix, y_pred_matrix, average='micro')

# 打印结果
print(f'Subset Accuracy: {subset_acc}')
print(f'Hamming Loss: {hamming}')
print(f'Jaccard Index: {jaccard}')
print(f'F1 Score (Micro): {f1}')
print(f'Precision (Micro): {precision}')
print(f'Recall (Micro): {recall}')
