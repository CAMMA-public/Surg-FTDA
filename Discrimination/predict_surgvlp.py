import sys
import os
import argparse
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as nnf
import clip  # installed from https://github.com/openai/CLIP
from transformers import GPT2Tokenizer

# Keep this import if CUDA(...) is defined in custom_types
from custom_types import *

# We only need ClipCaptionModel from train_surgvlp
from train_surgvlp import ClipCaptionModel

# We only need generate_beam and generate2 from gpt2_prefix_eval
from gpt2_prefix_eval import generate_beam, generate2


class Timer:
    """
    Measure inference time
    """
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
        std_syn = torch.std(torch.tensor(self.timings))
        return f"mean: {mean_syn:.2f} ms, std: {std_syn:.2f} ms"


def write_json(data, file_path):
    """Write data to a JSON file."""
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)


def make_preds(model: ClipCaptionModel, tokenizer, args):
    """
    :param model: ClipCaptionModel
    :param tokenizer: GPT2Tokenizer
    :param args: command line arguments
    """
    # Choose device
    device = CUDA(0)  # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load the appropriate CLIP model
    if args.is_rn:
        clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
        args.beam = True
    else:
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Optionally add modality offset
    modality_offset = None
    if args.add_modality_offset:
        with open(args.offset_dict_path, 'rb') as f:
            center_info = pickle.load(f)
        modality_offset = center_info['offset_to_add_in_inference'].to(device)

    # Optionally use a modality bridger
    map_to_text_space_using_modality_bridger = None
    if args.modality_bridger:
        from others.supervised_embedding_bridger import get_map_to_text_space_using_modality_bridger
        map_to_text_space_using_modality_bridger = get_map_to_text_space_using_modality_bridger()

    # Check dataset mode (0 or 5 in use; others for customized usage)
    if args.dataset_mode not in [0, 5, 7, 8, 1, 2, 3, 4, 6]:
        print("Wrong data mode")
        exit(1)

    # Just a timer if you want to measure inference time
    timer = Timer()
    print("ablation_dist is", args.ablation_dist)
    translations = []
    k = 0

    # Create the output directory
    if not os.path.exists(args.save_path_prefix):
        os.makedirs(args.save_path_prefix)
        print(f"Directory created: {args.save_path_prefix}")
    else:
        print(f"Directory already exists: {args.save_path_prefix}")

    # Load the data
    print("Data path:", args.data_path)
    with open(args.data_path, 'rb') as f:
        all_data = pickle.load(f)

    if args.dataset_mode == 0:
        print("Use image to inference")
        for ii, prefix in tqdm(enumerate(all_data['clip_embedding_text_dave'])):
            with torch.no_grad():
                timer.__enter__()
                prefix = torch.tensor(prefix).to(device).unsqueeze(0)

                if not args.dont_normalize_prefix:
                    prefix = prefix / prefix.norm(2, -1)
                if modality_offset is not None:
                    prefix = prefix + modality_offset
                if map_to_text_space_using_modality_bridger is not None:
                    prefix = map_to_text_space_using_modality_bridger(prefix)
                    prefix = prefix / prefix.norm(2, -1)

                prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)

            # Beam search or greedy
            if not args.beam:
                generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
            else:
                generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

            caption = all_data["captions"][ii]
            translation_pair = {"generated": generated_text_prefix, "reference": caption}
            translations.append(translation_pair)

            if (ii + 1) % 200 == 0:
                k += 1
                json_file_path = os.path.join(args.save_path_prefix, f'translations_{k}.json')
                translations_data = {"translations": translations}
                write_json(translations_data, json_file_path)
                print(f"Saved partial results to {json_file_path}")

            timer.__exit__()

        # Write final results
        final_json_file_path = os.path.join(args.save_path_prefix, 'translations_final.json')
        final_translations_data = {"translations": translations}
        write_json(final_translations_data, final_json_file_path)
        print(f"Done! Final results saved to {final_json_file_path}")

    elif args.dataset_mode == 5:
        print("Use text")
        for ii, prefix in tqdm(enumerate(all_data['clip_embedding_text_dave'])):
            with torch.no_grad():
                timer.__enter__()
                prefix = torch.tensor(prefix).to(device).unsqueeze(0)

                if not args.dont_normalize_prefix:
                    prefix = prefix / prefix.norm(2, -1)
                if modality_offset is not None:
                    prefix = prefix + modality_offset
                if map_to_text_space_using_modality_bridger is not None:
                    prefix = map_to_text_space_using_modality_bridger(prefix)
                    prefix = prefix / prefix.norm(2, -1)

                prefix_embed = model.clip_project(prefix).reshape(1, args.prefix_length, -1)

            if not args.beam:
                generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
            else:
                generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

            caption = all_data["captions"][ii]
            translation_pair = {"generated": generated_text_prefix, "reference": caption}
            translations.append(translation_pair)

            if (ii + 1) % 200 == 0:
                k += 1
                json_file_path = os.path.join(args.save_path_prefix, f'translations_{k}.json')
                translations_data = {"translations": translations}
                write_json(translations_data, json_file_path)
                print(f"Saved partial results to {json_file_path}")

            timer.__exit__()

        final_json_file_path = os.path.join(args.save_path_prefix, 'translations_final.json')
        final_translations_data = {"translations": translations}
        write_json(final_translations_data, final_json_file_path)
        print(f"Done! Final results saved to {final_json_file_path}")

    else:
        # For other dataset modes (1,2,3,4,6,7,8), keep the same logic if needed
        print("Dataset mode not implemented here or is custom.")


def main():
    print("Start...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("Loaded tokenizer")
    sys.stdout.flush()

    parser = argparse.ArgumentParser()
    # Checkpoint path
    parser.add_argument('--checkpoint', default='/home/ubuntu/meddataset/cholec_process/model_triplet/partial/10/coco_prefix-007.pt')
    # Output path is no longer used, remove if desired
    parser.add_argument('--out', default='')
    # Dataset mode
    parser.add_argument('--dataset_mode', type=int, default=0)
    parser.add_argument('--modality_bridger', action='store_true', default=False)
    parser.add_argument('--beam', action='store_true', default=True)
    parser.add_argument('--is_rn', action='store_true', default=True)
    parser.add_argument('--dont_normalize_prefix', action='store_true', default=False)
    parser.add_argument('--text_autoencoder', action='store_true', default=False)
    parser.add_argument('--add_modality_offset', action='store_true', default=False)
    parser.add_argument('--ablation_dist', action='store_true', default=False)
    parser.add_argument('--ablation_image_dist', action='store_true', default=False)
    parser.add_argument('--prefix_length', type=int, default=40)
    parser.add_argument('--prefix_length_clip', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=8)
    # Mapping type is not actually used in the final model loading but we keep it for reference
    parser.add_argument('--mapping_type', type=str, default='transformer_encoder', help='mlp/transformer_encoder/transformer_decoder')
    # Additional arguments for offsets, data path, save path
    parser.add_argument('--offset_dict_path', type=str, default='others/CLIP_embeddings_centers_info.pkl', help='path to offset dictionary if needed')
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/meddataset/cholec_process/triplet_test/embedding_nips3/image_embedding_triplet_original_test.pkl',
                        help='path to the pickled input data')
    parser.add_argument('--save_path_prefix', type=str, default='/home/ubuntu/meddataset/choelc_generate/triplet_test/partial/10',
                        help='where to save inference results')
    args = parser.parse_args()

    print(f"Beam search = {args.beam}")

    # Load the model
    model = ClipCaptionModel(
        prefix_length=40,  # keep it fixed to not break logic
        clip_length=args.prefix_length_clip,
        prefix_size=768,  # fixed as well (for surgvlp usage)
        num_layers=args.num_layers,
        mapping_type='transformer'  # keep it consistent
    )
    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=CUDA(0)))
    print("Loaded checkpoint:", args.checkpoint)
    print(f"modality_offset = {args.add_modality_offset}")

    # Make predictions
    make_preds(model, tokenizer, args)


if __name__ == '__main__':
    main()
