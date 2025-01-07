import json
import pickle
import torch
import surgvlp
from PIL import Image
from mmengine.config import Config
import argparse
import clip
from tqdm import tqdm
import os
import sys
from transformers import AutoTokenizer
import numpy as np
import torchvision.transforms as transforms

# Add custom module path
sys.path.append('/home/ubuntu/ClipMed/cvl-master')

def process_text(text, tokenizer_clinical, ixtoword, device):
    if isinstance(text, str):
        text = [text]

    processed_text_tensors = []
    for t in text:
        text_tensors = tokenizer_clinical(
            t,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=77,
        )
        text_tensors["sent"] = [
            ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
        ]
        processed_text_tensors.append(text_tensors)

    caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
    attention_mask = torch.stack(
        [x["attention_mask"] for x in processed_text_tensors]
    )
    token_type_ids = torch.stack(
        [x["token_type_ids"] for x in processed_text_tensors]
    )

    if len(text) == 1:
        caption_ids = caption_ids.squeeze(0).to(device)
        attention_mask = attention_mask.squeeze(0).to(device)
        token_type_ids = token_type_ids.squeeze(0).to(device)
    else:
        caption_ids = caption_ids.squeeze().to(device)
        attention_mask = attention_mask.squeeze().to(device)
        token_type_ids = token_type_ids.squeeze().to(device)

    cap_lens = [len([w for w in txt if not w.startswith("[")]) for txt in text]

    return {
        "input_ids": caption_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "cap_lens": cap_lens,
    }

def generate_image_embeddings(model, data, folder_path, transform_to_tensor, device, out_path, save_interval=10000):
    all_embeddings = []
    all_captions = []
    all_text_embeddings = []
    not_found = 0

    for i in tqdm(range(len(data)), desc="Processing Images"):
        d = data[i]
        img_id = d['image_name']
        image_path = os.path.join(folder_path, img_id)
        if not os.path.exists(image_path):
            not_found += 1
            continue

        try:
            frames = Image.open(image_path).convert("RGB")
            frames = transform_to_tensor(frames).to(device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            not_found += 1
            continue

        with torch.no_grad():
            C, H, W = frames.shape
            frames = frames.view(-1, C, H, W)
            image_features = model(frames, None, mode='video')['img_emb']  # (B*M*T, D)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prefix = torch.tensor([]).to(device)  # empty tensor

        all_embeddings.append(prefix)
        all_text_embeddings.append(image_features)
        all_captions.append(d)

        if (i + 1) % save_interval == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({
                    "clip_embedding": torch.cat(all_embeddings, dim=0),
                    "captions": all_captions,
                    'clip_embedding_text_dave': torch.cat(all_text_embeddings, dim=0)
                }, f)
            print(f"Saved {i+1} image embeddings.")

    # Final save
    with open(out_path, 'wb') as f:
        pickle.dump({
            "clip_embedding": torch.cat(all_embeddings, dim=0),
            "captions": all_captions,
            'clip_embedding_text_dave': torch.cat(all_text_embeddings, dim=0)
        }, f)

    print('Image Embedding Generation Done')
    print(f"{len(all_embeddings)} embeddings saved")
    print(f'Not found images = {not_found}')

def generate_text_embeddings(model, data, device, out_path, tokenizer_clinical, ixtoword, save_interval=10000):
    all_embeddings = []
    all_captions = []
    all_text_embeddings = []
    long_caps = 0

    for i in tqdm(range(len(data)), desc="Processing Texts"):
        d = data[i]
        caption = d["text"]
        if not caption:
            continue

        with torch.no_grad():
            try:
                caption_tokens = process_text(caption, tokenizer_clinical, ixtoword, device)
                #caption_tokens = surgvlp.tokenize(caption, device=device)
            except Exception as e:
                # Attempt to truncate long captions
                try:
                    caption_tokens = process_text(caption[:100], tokenizer_clinical, ixtoword, device)
                    #caption_tokens = surgvlp.tokenize(caption[:100], device=device)
                    long_caps += 1
                    print(f'Long caption truncated: {caption[:100]}')
                except Exception as ex:
                    print(f"Error processing caption: {ex}")
                    continue

            caption_embedding = model(None, caption_tokens, mode='text')['text_emb']

        prefix = torch.tensor([]).to(device)  # empty tensor
        all_embeddings.append(prefix)
        all_text_embeddings.append(caption_embedding)
        all_captions.append(d)

        if (i + 1) % save_interval == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({
                    "clip_embedding": torch.cat(all_embeddings, dim=0),
                    "captions": all_captions,
                    'clip_embedding_text_dave': torch.cat(all_text_embeddings, dim=0)
                }, f)
            print(f"Saved {i+1} text embeddings.")

    # Final save
    with open(out_path, 'wb') as f:
        pickle.dump({
            "clip_embedding": torch.cat(all_embeddings, dim=0),
            "captions": all_captions,
            'clip_embedding_text_dave': torch.cat(all_text_embeddings, dim=0)
        }, f)

    print('Text Embedding Generation Done')
    print(f"{len(all_embeddings)} embeddings saved")
    print(f'Long captions truncated: {long_caps}')

def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def main(args):
    print("Starting script...")
    print("Using Python executable:", sys.executable)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load configuration
    configs = Config.fromfile(args.config_file)['config']
    print("Configuration loaded.")

    # Load model
    model, preprocess = surgvlp.load(configs.model_config, device=device, pretrain=args.model_path)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Load data
    with open(args.annotations_path, 'r') as f:
        data = json.load(f)
    print(f"{len(data)} captions loaded from JSON.")

    # Define transformation
    transform_to_tensor = transforms.ToTensor()

    if args.mode in ['image', 'both']:
        print("Generating Image Embeddings...")
        ensure_dir(args.out_image_out_path)
        generate_image_embeddings(
            model=model,
            data=data,
            folder_path=args.image_folder_path,
            transform_to_tensor=transform_to_tensor,
            device=device,
            out_path=args.out_image_out_path,
            save_interval=args.save_interval
        )

    if args.mode in ['text', 'both']:
        print("Generating Text Embeddings...")
        ensure_dir(args.out_text_out_path)
        tokenizer_clinical = AutoTokenizer.from_pretrained('/home/ubuntu/ClipMed/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000')
        ixtoword = {v: k for k, v in tokenizer_clinical.get_vocab().items()}
        generate_text_embeddings(
            model=model,
            data=data,
            device=device,
            out_path=args.out_text_out_path,
            tokenizer_clinical=tokenizer_clinical,
            ixtoword=ixtoword,
            save_interval=args.save_interval
        )

    print("All tasks completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Image and/or Text Embeddings")

    parser.add_argument(
        '--mode',
        type=str,
        choices=['image', 'text', 'both'],
        default='both',
        help='Mode to run the script: image, text, or both.'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='/home/ubuntu/Surg-FTDA_backup/config_surgvlp.py',
        help='Path to the config file.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/ubuntu/Surg-FTDA_backup/ClipMed/nips_v3.tar',
        help='Path to the pretrained model.'
    )
    parser.add_argument(
        '--annotations_path',
        type=str,
        default='/home/ubuntu/meddataset/meddatasetnew/train/train_file.json',
        help='Path to the annotations JSON file.'
    )
    parser.add_argument(
        '--image_folder_path',
        type=str,
        default='/home/ubuntu/meddataset/jianan/svl_frames_phaseonly',
        help='Path to the folder containing images.'
    )
    parser.add_argument(
        '--out_image_out_path',
        type=str,
        default='/home/ubuntu/meddataset/meddatasetnew/train_test/embedding_step/embedding_nipsv3_v2/embedding_image_train.pkl',
        help='Output path for image embeddings.'
    )
    parser.add_argument(
        '--out_text_out_path',
        type=str,
        default='/home/ubuntu/meddataset/meddatasetnew/train_test/embedding_step/embedding_nipsv3_v2/embedding_text_train.pkl',
        help='Output path for text embeddings.'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=10000,
        help='Interval for saving embeddings.'
    )

    args = parser.parse_args()
    main(args)
