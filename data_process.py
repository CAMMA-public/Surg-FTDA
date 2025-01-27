import json

# 读取原始JSON数据
# input_file = '/home/ubuntu/meddataset/cholec_process/dataset_opensource/triplet/triplet_test.json'  # 输入JSON文件路径
# output_file = '/home/ubuntu/meddataset/cholec_process/dataset_opensource/triplet/process/triplet_test.json'  # 输出JSON文件路径

# with open(input_file, 'r') as f:
#     data = json.load(f)

# # 修改image_path，仅保留相对路径
# for item in data:
#     item['image_path'] = '/'.join(item['image_path'].split('/')[-3:])

# # 将修改后的数据写入新的JSON文件
# with open(output_file, 'w') as f:
#     json.dump(data, f, indent=4)

# print(f"Modified JSON file saved as {output_file}")

import json

def update_json_with_absolute_path(input_file, output_file, absolute_data_path):
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 更新image_path为绝对路径
    for item in data:
        item['image_path'] = f"{absolute_data_path}/{item['image_path']}"

    # 保存更新后的数据
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Updated JSON file saved as {output_file}")

# 示例调用
input_file = '/home/ubuntu/meddataset/cholec_process/dataset_opensource/triplet/process/test_triplet.json'  # 输入JSON文件路径
output_file = '/home/ubuntu/meddataset/cholec_process/dataset_opensource/triplet/process/triplet_test_test.json'  # 输出JSON文件路径
absolute_data_path = '/home/ubuntu/meddataset/cholec/unistra_file/CholecT50-challenge-train'
update_json_with_absolute_path(input_file, output_file, absolute_data_path)
