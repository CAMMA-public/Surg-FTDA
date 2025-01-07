import argparse
import json
import os
import pickle

import torch
import clip
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms


def ensure_dir(path: str):
    """
    Creates the directory for the given file path if it does not exist.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def generate_image_embeddings(model, preprocess, data, folder_path, device, out_path, save_interval=10000):
    """
    Generates image embeddings using CLIP, then saves them (along with the relevant metadata).
    Both 'clip_embedding' (prefix, which is empty) and 'clip_embedding_text_dave' (the actual embeddings)
    are stored in the final dictionary.
    """
    all_embeddings = []          # Will hold the 'prefix' (empty)
    all_text_embeddings = []     # Will hold the actual image embeddings
    all_captions = []
    not_found = 0

    for i in tqdm(range(len(data)), desc="Processing Images"):
        d = data[i]
        # Change "image_name" to the correct key in your data
        img_id = d.get("image_path", None)
        if img_id is None:
            continue

        image_path = os.path.join(folder_path, img_id)
        if not os.path.exists(image_path):
            not_found += 1
            continue

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            # Preprocess image (CLIP's default preprocessing)
            image_input = preprocess(image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            not_found += 1
            continue

        with torch.no_grad():
            # Encode image with CLIP
            image_features = model.encode_image(image_input).cpu()
            # You can normalize if you want: image_features /= image_features.norm(dim=-1, keepdim=True)
            # Here we keep it unnormalized to mirror your original code's pattern.

        # Prefix is empty (same style as the SurgVLP code).
        prefix = torch.tensor([])

        # Collect embeddings
        all_embeddings.append(prefix)
        all_text_embeddings.append(image_features)
        all_captions.append(d)

        # Save intermediate results every 'save_interval' items
        if (i + 1) % save_interval == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({
                    "clip_embedding": torch.cat(all_embeddings, dim=0),
                    "captions": all_captions,
                    # We store all embeddings (image ones in this case) in 'clip_embedding_text_dave' as well
                    "clip_embedding_text_dave": torch.cat(all_text_embeddings, dim=0)
                }, f)
            print(f"Saved {i+1} image embeddings so far.")

    # Final save
    with open(out_path, 'wb') as f:
        pickle.dump({
            "clip_embedding": torch.cat(all_embeddings, dim=0),
            "captions": all_captions,
            "clip_embedding_text_dave": torch.cat(all_text_embeddings, dim=0)
        }, f)

    print("Image Embedding Generation Complete")
    print(f"{len(all_embeddings)} image embeddings saved.")
    print(f"Images not found: {not_found}")


def generate_text_embeddings(model, data, device, out_path, save_interval=10000):
    """
    Generates text embeddings using CLIP, then saves them (along with the relevant metadata).
    Both 'clip_embedding' (prefix, which is empty) and 'clip_embedding_text_dave' (the actual embeddings)
    are stored in the final dictionary.
    """
    all_embeddings = []          # Will hold the 'prefix' (empty)
    all_text_embeddings = []     # Will hold the actual text embeddings
    all_captions = []
    long_caps = 0

    for i in tqdm(range(len(data)), desc="Processing Texts"):
        d = data[i]
        # Change "text" or "step" to the correct key in your data
        caption = None
        if "triplets" in d:
            caption = d["triplets"]
        elif "phases" in d:
            caption = d["phases"]

        with torch.no_grad():
            try:
                # Tokenize text
                text_tokens = clip.tokenize(caption).to(device)
            except Exception:
                # If text is too long, truncate
                text_tokens = clip.tokenize(caption[:100]).to(device)
                long_caps += 1
                print(f"Truncated long caption: {caption}")

            # Encode text with CLIP
            text_features = model.encode_text(text_tokens).cpu()
            # You can normalize if you want: text_features /= text_features.norm(dim=-1, keepdim=True)

        # Prefix is empty (same style as the SurgVLP code).
        prefix = torch.tensor([])

        # Collect embeddings
        all_embeddings.append(prefix)
        all_text_embeddings.append(text_features)
        all_captions.append(d)

        # Save intermediate results every 'save_interval' items
        if (i + 1) % save_interval == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({
                    "clip_embedding": torch.cat(all_embeddings, dim=0),
                    "captions": all_captions,
                    "clip_embedding_text_dave": torch.cat(all_text_embeddings, dim=0)
                }, f)
            print(f"Saved {i+1} text embeddings so far.")

    # Final save
    with open(out_path, 'wb') as f:
        pickle.dump({
            "clip_embedding": torch.cat(all_embeddings, dim=0),
            "captions": all_captions,
            "clip_embedding_text_dave": torch.cat(all_text_embeddings, dim=0)
        }, f)

    print("Text Embedding Generation Complete")
    print(f"{len(all_embeddings)} text embeddings saved.")
    print(f"Number of truncated long captions: {long_caps}")



def main():
    """
    Main function to parse arguments, load CLIP, and generate image/text embeddings as requested.
    """
    parser = argparse.ArgumentParser(description="Generate Image and/or Text Embeddings with CLIP")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['image', 'text', 'both'],
        default='both',
        help='Mode to run the script: image, text, or both.'
    )
    parser.add_argument(
        '--clip_model_type',
        type=str,
        default='RN50x4',
        help='CLIP model type, e.g. RN50, RN101, RN50x4, ViT-B/32, etc.'
    )
    parser.add_argument(
        '--annotations_path',
        type=str,
        default='/home/ubuntu/meddataset/cholec_process/combine_train/combine_phase_triplet.json',
        help='Path to the annotations JSON file.'
    )
    parser.add_argument(
        '--image_folder_path',
        type=str,
        default='/home/ubuntu/meddataset/cholec_process/data',
        help='Path to the folder containing images.'
    )
    parser.add_argument(
        '--out_image_path',
        type=str,
        default='/home/ubuntu/meddataset/meddatasetnew/train_test/embedding_step/embedding_clip/embedding_image_train.pkl',
        help='Output path for image embeddings.'
    )
    parser.add_argument(
        '--out_text_path',
        type=str,
        default='/home/ubuntu/meddataset/meddatasetnew/train_test/embedding_step/embedding_clip/embedding_text_train.pkl',
        help='Output path for text embeddings.'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=10000,
        help='Interval for saving intermediate embeddings.'
    )
    args = parser.parse_args()

    # Print the resolved paths
    print(f"Annotations path: {args.annotations_path}")
    print(f"Image folder path: {args.image_folder_path}")
    print(f"Output image path: {args.out_image_path}")
    print(f"Output text path: {args.out_text_path}")

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model '{args.clip_model_type}' on device: {device}")
    clip_model, preprocess = clip.load(args.clip_model_type, device=device, jit=False)
    clip_model.eval()

    # Read JSON data
    with open(args.annotations_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {args.annotations_path}")

    # Ensure output directories exist
    ensure_dir(args.out_image_path)
    ensure_dir(args.out_text_path)

    # Generate embeddings
    if args.mode in ['image', 'both']:
        print("Generating image embeddings...")
        generate_image_embeddings(
            model=clip_model,
            preprocess=preprocess,
            data=data,
            folder_path=args.image_folder_path,
            device=device,
            out_path=args.out_image_path,
            save_interval=args.save_interval
        )

    if args.mode in ['text', 'both']:
        print("Generating text embeddings...")
        generate_text_embeddings(
            model=clip_model,
            data=data,
            device=device,
            out_path=args.out_text_path,
            save_interval=args.save_interval
        )

    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
