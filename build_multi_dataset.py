"""
build_multi_dataset.py — Modular extraction for multi-model scaling.
Supports Gemma-2-2b, Gemma-2-9b, Llama-3-8B, etc.
"""

import os
import torch
import torch.nn.functional as F
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# ── Extraction Logic ──────────────────────────────────────────────────────────

def extract_hidden_states(texts, tokenizer, model, device, max_length=64, batch_size=32):
    all_hidden = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Take final residual stream (B, L, D) -> last token (B, D)
            hidden_res = outputs.hidden_states[-1][:, -1, :]
            all_hidden.append(hidden_res.cpu().float())
    return torch.cat(all_hidden, dim=0)

def build_knn_sets(hidden: torch.Tensor, texts: list[str], set_size: int, n_sets: int):
    N, D = hidden.shape
    normed = F.normalize(hidden, dim=-1)
    sim = normed @ normed.T
    n_anchors = min(n_sets, N)
    anchor_idx = torch.randperm(N)[:n_anchors]
    sets = []
    text_sets = []
    for a in anchor_idx:
        top_k = sim[a].argsort(descending=True)[:set_size]
        sets.append(hidden[top_k])
        text_sets.append([texts[i.item()] for i in top_k])
    return torch.stack(sets, dim=0), text_sets

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the dataset")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit for VRAM saving")
    parser.add_argument("--set_size", type=int, default=32)
    parser.add_argument("--texts_per_class", type=int, default=1024)
    parser.add_argument("--sets_per_class", type=int, default=512)
    parser.add_argument("--hf_token", type=str, help="HuggingFace Token")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True) if os.path.dirname(args.output_path) else None

    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model_id}...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(args.model_id, token=args.hf_token, quantization_config=bnb_config, device_map="auto")
    else:
        model = AutoModel.from_pretrained(args.model_id, token=args.hf_token, torch_dtype=dtype, device_map="auto")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news", split="train")

    grouped_texts = {}
    for item in dataset:
        lbl = item["label"]
        if lbl not in grouped_texts: grouped_texts[lbl] = []
        if len(grouped_texts[lbl]) < args.texts_per_class:
            grouped_texts[lbl].append(item["text"])
        if all(len(g) >= args.texts_per_class for g in grouped_texts.values()) and len(grouped_texts) == 4:
            break

    all_class_sets = []
    all_class_texts = []
    for lbl, texts in sorted(grouped_texts.items()):
        print(f"Class {lbl}: extracting hidden states...")
        hidden = extract_hidden_states(texts, tokenizer, model, device)
        class_sets, class_text_sets = build_knn_sets(hidden, texts, args.set_size, args.sets_per_class)
        all_class_sets.append(class_sets)
        all_class_texts.extend(class_text_sets)

    final_dataset = torch.cat(all_class_sets, dim=0)
    perm = torch.randperm(len(final_dataset))
    final_dataset = final_dataset[perm]
    final_texts = [all_class_texts[i.item()] for i in perm]

    save_dict = {"hidden": final_dataset, "texts": final_texts }
    torch.save(save_dict, args.output_path)
    print(f"Dataset saved to {args.output_path}. Shape: {final_dataset.shape}")

if __name__ == "__main__":
    main()
