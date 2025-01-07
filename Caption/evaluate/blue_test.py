import json
import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from bert_score import score as bert_score

# 加载您的JSON数据
json_file_path = '/home/ubuntu/meddataset/meddatasetnew/all_generate/val/nips3v2/inference_kmeans500v2/translations_final.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)


candidates = []
references = []
meteor_scores = []
bert_candidates = []
bert_references = []

for item in data['translations']:
    generated_text = item['generated']
    reference_text = item['reference']["text"]

    generated_tokens = generated_text.split()
    reference_tokens = [reference_text.split()]  
    candidates.append(generated_tokens)
    references.append(reference_tokens)

    meteor_score_value = meteor_score([reference_text.split()], generated_text.split())
    meteor_scores.append(meteor_score_value)


    bert_candidates.append(generated_text)
    bert_references.append(reference_text)


smooth_fn = SmoothingFunction().method1
score_bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
score_bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

print(f"Corpus BLEU-1 score: {score_bleu1}")
print(f"Corpus BLEU-4 score: {score_bleu4}")


average_meteor = sum(meteor_scores) / len(meteor_scores)
print(f"Average METEOR score: {average_meteor}")


gts = {}  # groud truth
res = {}  # generated

for idx, item in enumerate(data["translations"]):
    img_id = str(idx)  
    gts[img_id] = [item["reference"]["text"]]
    res[img_id] = [item["generated"]]


scorer = Cider()
score_cider, scores_cider = scorer.compute_score(gts, res)
print(f"CIDEr Score: {score_cider}")

# 计算BERTScore
P, R, F1 = bert_score(bert_candidates, bert_references, lang='en', rescale_with_baseline=True)
average_bertscore = F1.mean().item()
print(f"Average BERTScore F1: {average_bertscore}")
