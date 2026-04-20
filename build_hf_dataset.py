"""
build_hf_dataset.py — Set-ConCA real-data extraction from Phi-2 on AG News.

Key design decisions aligned with Set-ConCA paper (Section 2):
  - Each set X = {x_1, ..., x_m} must represent SEMANTICALLY RELATED inputs
    (paraphrases, local neighbourhoods), NOT arbitrary consecutive texts.
  - We build sets using nearest-neighbour search in the hidden-state space:
    for each anchor, its 7 nearest neighbours (by cosine similarity) form a set.
  - This ensures intra-set variance << inter-set variance, which is the
    assumption underlying the Subset Consistency objective (Section 4.5).
"""

import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAME      = "ag_news"
MODEL_NAME        = "google/gemma-2-2b"
SET_SIZE          = 32       # Master set size for S=8,16,32 sweep
MAX_TEXTS_PER_CLASS = 1024   # texts to extract per class
N_SETS_PER_CLASS  = 512      # anchor sets selected per class
MAX_SEQ_LEN       = 64       # truncation for speed
BATCH_SIZE        = 32
OUT_PATH          = "data/gemma_sweep_dataset.pt"
# ──────────────────────────────────────────────────────────────────────────────


def extract_hidden_states(texts, tokenizer, model, device, max_length=MAX_SEQ_LEN, batch_size=BATCH_SIZE):
    """Extract last-token hidden states from the final residual stream (Layer 25)."""
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
            # Gemma-2-2b has 26 layers. hidden_states index 26 is output of block 25.
            # (B, L, D) -> last token (B, D)
            hidden_res = outputs.hidden_states[-1][:, -1, :]
            all_hidden.append(hidden_res.cpu().float())
    return torch.cat(all_hidden, dim=0)  # (N, D)


def build_knn_sets(hidden: torch.Tensor, texts: list[str], set_size: int, n_sets: int) -> tuple[torch.Tensor, list[list[str]]]:
    """
    Build representation sets using cosine k-NN within a class.

    For each anchor (sampled without replacement), find the (set_size - 1)
    nearest neighbours by cosine similarity. This guarantees semantically
    coherent sets rather than arbitrary sequential groupings.

    Returns: 
        sets: (n_sets, set_size, D)
        text_sets: list (len=n_sets) of lists (len=set_size) of strings
    """
    N, D = hidden.shape
    # L2-normalise for cosine sim via dot product
    normed = F.normalize(hidden, dim=-1)           # (N, D)
    # Full pairwise cosine similarity matrix — feasible for N ≤ 4096
    sim = normed @ normed.T                         # (N, N)

    # Sample anchors (without replacement)
    n_anchors = min(n_sets, N)
    anchor_idx = torch.randperm(N)[:n_anchors]

    sets = []
    text_sets = []
    for a in anchor_idx:
        # Sorted by descending similarity; exclude the anchor itself (index 0)
        top_k = sim[a].argsort(descending=True)[:set_size]  # includes self at 0
        sets.append(hidden[top_k])  # (set_size, D)
        text_sets.append([texts[i.item()] for i in top_k])

    return torch.stack(sets, dim=0), text_sets


def validate_dataset(data: torch.Tensor) -> None:
    """
    Verify that representation sets are semantically coherent and print a health report.

    For Set-ConCA, GOOD data satisfies:
      intra-set variance  <  inter-set variance
    i.e. elements within a set are more similar to each other than sets are to each
    other on average.  This matches the paper (Sec 2): sets should be paraphrases or
    local neighbourhoods, not arbitrary groupings.

    Ratio = intra / inter:
      < 0.5  → strong coherence  (elements tightly clustered within sets)
      0.5-1  → moderate coherence (kNN grouping is working)
      > 1.0  → sets are NOT coherent (sequential chunk problem — fix build script!)
    """
    N, S, D = data.shape
    intra_var = data.var(dim=1).mean().item()         # variance within each set
    set_means = data.mean(dim=1)                      # (N, D)
    inter_var = set_means.var(dim=0).mean().item()    # variance of set means

    print("\n-- Dataset Validation -------------------------------------------")
    print(f"  Shape  : {tuple(data.shape)}")
    print(f"  dtype  : {data.dtype}")
    print(f"  Min/Max: {data.min():.3f} / {data.max():.3f}")
    print(f"  NaN    : {torch.isnan(data).any().item()}")
    print(f"  Inf    : {torch.isinf(data).any().item()}")
    print(f"  Intra-set variance (mean): {intra_var:.4f}")
    print(f"  Inter-set variance (mean): {inter_var:.4f}")
    ratio = intra_var / (inter_var + 1e-8)
    print(f"  Intra/Inter ratio        : {ratio:.4f}  (want < 1.0)")
    if ratio < 0.5:
        print("  PASS  Strong coherence - sets are tight paraphrase clusters.")
    elif ratio < 1.0:
        print("  PASS  Moderate coherence - kNN grouping is working.")
    else:
        print("  FAIL  Intra >= Inter - sets are NOT coherent. Check build script!")

    # Cosine similarity summary (fast: sample 200 sets)
    import torch.nn.functional as F
    N, S, D = data.shape
    n_sample = min(200, N)
    sample = data[:n_sample]
    normed = F.normalize(sample.float(), dim=-1)  # (n_sample, S, D)
    # Intra: mean pairwise cosine sim within each set (exclude diagonal)
    gram = torch.bmm(normed, normed.transpose(1, 2))  # (n_sample, S, S)
    mask = ~torch.eye(S, dtype=torch.bool).unsqueeze(0)
    intra_cos = gram[mask.expand_as(gram)].mean().item()
    # Inter: cosine sim between set means
    set_means = normed.mean(dim=1)  # (n_sample, D)
    set_means = F.normalize(set_means, dim=-1)
    inter_gram = set_means @ set_means.T  # (n_sample, n_sample)
    off_diag = inter_gram[~torch.eye(n_sample, dtype=torch.bool)]
    inter_cos = off_diag.mean().item()
    print(f"  Intra-set cosine sim     : {intra_cos:.4f}  (want > inter)")
    print(f"  Inter-set cosine sim     : {inter_cos:.4f}")
    cos_ratio = intra_cos / (inter_cos + 1e-8)
    print(f"  Cosine intra/inter ratio : {cos_ratio:.4f}  (want > 1.0)")
    if intra_cos > inter_cos:
        print("  PASS  Sets are cosine-similar within, dissimilar across.")
    else:
        print("  WARN  Sets not more self-similar than cross-set pairs.")
    print("-----------------------------------------------------------------------\n")

    assert not torch.isnan(data).any(), "Dataset contains NaN!"
    assert not torch.isinf(data).any(), "Dataset contains Inf!"


def main():
    print(f"Loading dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split="train")

    print(f"Loading model '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 if supported, else float16
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModel.from_pretrained(MODEL_NAME, token=True, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model on {device} (dtype={dtype}).")

    # ── Collect texts per class ───────────────────────────────────────────────
    grouped_texts: dict[int, list[str]] = {}
    for item in dataset:
        lbl = item["label"]
        if lbl not in grouped_texts:
            grouped_texts[lbl] = []
        if len(grouped_texts[lbl]) < MAX_TEXTS_PER_CLASS:
            grouped_texts[lbl].append(item["text"])
        if all(len(g) >= MAX_TEXTS_PER_CLASS for g in grouped_texts.values()) and len(grouped_texts) == 4:
            break

    print(f"Collected texts: { {k: len(v) for k, v in grouped_texts.items()} }")

    # ── Extract hidden states & build kNN sets per class ─────────────────────
    all_class_sets = []
    all_class_texts = []
    for lbl, texts in sorted(grouped_texts.items()):
        print(f"\nClass {lbl}: extracting {len(texts)} hidden states...")
        hidden = extract_hidden_states(texts, tokenizer, model, device)
        print(f"  Hidden shape: {hidden.shape}")

        print(f"  Building kNN sets (set_size={SET_SIZE}, n_sets={N_SETS_PER_CLASS})...")
        class_sets, class_text_sets = build_knn_sets(hidden, texts, set_size=SET_SIZE, n_sets=N_SETS_PER_CLASS)
        print(f"  Class sets shape: {class_sets.shape}")
        all_class_sets.append(class_sets)
        all_class_texts.extend(class_text_sets)

    # ── Assemble & save ───────────────────────────────────────────────────────
    final_dataset = torch.cat(all_class_sets, dim=0)  # (N_total, set_size, D)
    
    # Shuffle across classes
    perm = torch.randperm(len(final_dataset))
    final_dataset = final_dataset[perm]
    final_texts = [all_class_texts[i.item()] for i in perm]

    validate_dataset(final_dataset)

    os.makedirs("data", exist_ok=True)
    
    # Save both embeddings and their corresponding texts
    save_dict = {
        "hidden": final_dataset,
        "texts": final_texts
    }
    torch.save(save_dict, OUT_PATH)
    print(f"Saved -> {OUT_PATH}  shape={tuple(final_dataset.shape)} with text pairs.")


if __name__ == "__main__":
    main()
