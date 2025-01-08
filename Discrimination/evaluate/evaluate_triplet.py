import json
with open('/home/ubuntu/meddataset/cholec/unistra_file/CholecT50-challenge-train/dict/triplet.txt', 'r') as file:
    lines = file.readlines()

def count_unique_triplets(lines):
    triplet_set = set()
    
    for line in lines:

        triplet_str = line.strip().split(':', 1)[-1]
        triplets = triplet_str.split(',')  
        formatted_triplet = f"({triplets[0]},{triplets[1]},{triplets[2]})"

        triplet_set.add(formatted_triplet)
    
    return len(triplet_set), triplet_set


unique_triplet_count, unique_triplets = count_unique_triplets(lines)
unique_triplets_list = list(unique_triplets)
print(len(unique_triplets_list))

file_path = "/home/ubuntu/meddataset/choelc_generate/combine/v2/triplet/inference_kmeans500v2/translations_final.json"


generated_triplets = []
reference_triplets = []

with open(file_path, 'r') as file:
    data = json.load(file)


    for item in data['translations']:

        generated_str = item['generated'].replace("triplet: ", "").strip()


        generated_list = [triplet.strip() for triplet in generated_str.split("),") if triplet]
        generated_list = [triplet + ")" if not triplet.endswith(")") else triplet for triplet in generated_list]  # 确保每个三元组以 ")" 结尾
        generated_triplets.append(generated_list)


        reference_str = item['reference']['triplets'].replace("triplet: ", "").strip()


        reference_list = [triplet.strip() for triplet in reference_str.split("),") if triplet]
        reference_list = [triplet + ")" if not triplet.endswith(")") else triplet for triplet in reference_list]  # 确保每个三元组以 ")" 结尾
        reference_triplets.append(reference_list)



from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, f1_score, precision_score, recall_score


triplet_to_index = {triplet: idx for idx, triplet in enumerate(unique_triplets_list)}

y_true_triplets_list = reference_triplets
y_pred_triplets_list = generated_triplets


y_true_matrix = []
y_pred_matrix = []

for y_true_triplets in y_true_triplets_list:
    y_true = [0] * len(unique_triplets_list) 
    if "no triplet" not in y_true_triplets:
        for triplet in y_true_triplets:
            if triplet in triplet_to_index:
                y_true[triplet_to_index[triplet]] = 1
    y_true_matrix.append(y_true)

for y_pred_triplets in y_pred_triplets_list:
    y_pred = [0] * len(unique_triplets_list)  
    if "no triplet" not in y_pred_triplets:
        for triplet in y_pred_triplets:
            if triplet in triplet_to_index:
                y_pred[triplet_to_index[triplet]] = 1
    y_pred_matrix.append(y_pred)


subset_acc = accuracy_score(y_true_matrix, y_pred_matrix)
hamming = hamming_loss(y_true_matrix, y_pred_matrix)
jaccard = jaccard_score(y_true_matrix, y_pred_matrix, average='samples',zero_division=1)
f1 = f1_score(y_true_matrix, y_pred_matrix, average='micro')
precision = precision_score(y_true_matrix, y_pred_matrix, average='micro')
recall = recall_score(y_true_matrix, y_pred_matrix, average='micro')


print(f'Subset Accuracy: {subset_acc}')
print(f'Hamming Loss: {hamming}')
print(f'Jaccard Index: {jaccard}')
print(f'F1 Score (Micro): {f1}')
print(f'Precision (Micro): {precision}')
print(f'Recall (Micro): {recall}')
