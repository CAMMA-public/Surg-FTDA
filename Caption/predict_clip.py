import sys
import os
import argparse
import json
import pickle
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as nnf
import matplotlib.pyplot as plt

sys.path.append("/home/amir/projects/CLIP")

from transformers import GPT2Tokenizer
from PIL import Image
import clip  # installed from https://github.com/openai/CLIP

from train_clip import ClipCaptionModel
from gpt2_prefix_eval import generate_beam, generate2


class Timer:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.timings = []

    def __enter__(self):
        self.starter.record()
        return self

    def __exit__(self, *args):
        self.ender.record()
        torch.cuda.synchronize()
        interval = self.starter.elapsed_time(self.ender)
        self.timings.append(interval)
        self.sum += interval
        self.count += 1

    def __str__(self):
        mean_syn = self.sum / self.count
        std_syn = np.std(self.timings)
        return f"mean: {mean_syn:.2f} ms, std: {std_syn:.2f} ms"


def write_json(data, file_path):
    """将数据写入JSON文件的辅助函数。"""
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)


def make_preds(model, tokenizer, args):
    """
    :param model: ClipCaptionModel
    :param tokenizer: GPT2Tokenizer

    """

    if args.dataset_mode not in [0, 5]:
        print("Wrong data mode")
        exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()


    if args.is_rn:
        clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
        args.beam = True
    else:
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


    if args.add_modality_offset:

        offset_dict_path = args.offset_dict_path  
        with open(offset_dict_path, 'rb') as f:
            center_info = pickle.load(f)
        modality_offset = center_info['offset_to_add_in_inference'].to(device)
    else:
        modality_offset = None

    if args.modality_bridger:
        from others.supervised_embedding_bridger import get_map_to_text_space_using_modality_bridger
        map_to_text_space_using_modality_bridger = get_map_to_text_space_using_modality_bridger()
    else:
        map_to_text_space_using_modality_bridger = None

    # embeddings = model.gpt.get_input_embeddings().weight.data
    # embeddings = nnf.normalize(embeddings, 2, 1)

    timer = Timer()
    print("ablation_dist is", args.ablation_dist)
    translations = []
    k = 0

    if not os.path.exists(args.save_path_prefix):
        os.makedirs(args.save_path_prefix)
        print(f"Directory created: {args.save_path_prefix}")
    else:
        print(f"Directory already exists: {args.save_path_prefix}")

    print("Use data path:", args.data_path)
    with open(args.data_path, 'rb') as f:
        all_data = pickle.load(f)

    for ii, prefix in tqdm(enumerate(all_data['clip_embedding_text_dave'])):
        with torch.no_grad():
            timer.__enter__()
            prefix = prefix.float().to(device).unsqueeze(0)

            if not args.dont_normalize_prefix:
                prefix = prefix / prefix.norm(2, -1)
            if modality_offset is not None:
                prefix = prefix + modality_offset
            if map_to_text_space_using_modality_bridger is not None:
                prefix = map_to_text_space_using_modality_bridger(prefix)
                prefix = prefix / prefix.norm(2, -1)

            prefix_embed = model.clip_project(prefix).reshape(
                1, args.prefix_length, -1
            )

        # beam 搜索 or greedy
        if not args.beam:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

        caption = all_data["captions"][ii]
        translation_pair = {"generated": generated_text_prefix, "reference": caption}
        translations.append(translation_pair)

        # 每 200 个保存一次
        if (ii + 1) % 200 == 0:
            k += 1
            json_file_path = os.path.join(args.save_path_prefix, f'translations_{k}.json')
            print(json_file_path)
            translations_data = {"translations": translations}
            write_json(translations_data, json_file_path)
            print(f"write in {json_file_path}")

        timer.__exit__()


    final_json_file_path = os.path.join(args.save_path_prefix, 'translations_final.json')
    final_translations_data = {"translations": translations}
    write_json(final_translations_data, final_json_file_path)
    print(f"Final JSON File write in {final_json_file_path}")


def main():
    print('start....')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print('loaded tokenizer')
    sys.stdout.flush()

    parser = argparse.ArgumentParser()


    parser.add_argument('--checkpoint', default='/home/ubuntu/meddataset/meddatasetnew/model/model_clip/noise0text/coco_prefix-009.pt')
    parser.add_argument('--dataset_mode', type=int, default=0,
                        help='0 for val with image, 5 for val text only')
    parser.add_argument('--beam', action='store_true', default=True, help='whether to use beam search')
    parser.add_argument('--is_rn', action='store_true', default=True, help='use RN50x4 CLIP model or not')
    parser.add_argument('--dont_normalize_prefix', action='store_true', default=False)
    parser.add_argument('--add_modality_offset', action='store_true', default=False)
    parser.add_argument('--modality_bridger', action='store_true', default=False)
    parser.add_argument('--ablation_dist', action='store_true', default=False, help='just for debugging print')
    parser.add_argument('--prefix_length', type=int, default=40)
    parser.add_argument('--prefix_length_clip', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--prefix_size', type=int, default=640, help='transformer input dim after clip projection')
    parser.add_argument('--offset_dict_path', type=str, default='others/CLIP_embeddings_centers_info.pkl',
                        help='path to offset dictionary if needed')
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/meddataset/meddatasetnew/val/embedding_text/clip/val_clip_text.pkl',
                        help='pickled file containing text and embeddings')
    parser.add_argument('--save_path_prefix', type=str,
                        default='/home/ubuntu/meddataset/meddatasetnew/all_generate/val/clip/reconstruction',
                        help='where to save output JSON files')

    args = parser.parse_args()
    print(f'beam search = {args.beam}')

    # 加载 checkpoint
    model = ClipCaptionModel(
        prefix_length=40,                      # 保持与原逻辑一致，不改写成 args.prefix_length
        clip_length=args.prefix_length_clip,
        prefix_size=args.prefix_size,
        num_layers=args.num_layers,
        mapping_type='transformer'             # 固定使用 'transformer'，不再从命令行传入
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print("Loaded checkpoint:", args.checkpoint)
    print(f'modality_offset = {args.add_modality_offset}')

    # 推理
    make_preds(model, tokenizer, args)


if __name__ == '__main__':
    main()
