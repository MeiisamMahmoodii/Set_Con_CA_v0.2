import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_utils.benchmark_builder import (
    BENCH_DIR,
    build_records,
    sanitize_model_name,
    save_benchmark_manifest,
    save_record_manifest,
)
from dataset_utils.extract_activations import ActivationExtractor


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


MODEL_SPECS = {
    "Llama-3.2-1B-Instruct": {"hf_name": "meta-llama/Llama-3.2-1B-Instruct", "relative_depth": 0.6},
    "Llama-3.2-3B-Instruct": {"hf_name": "meta-llama/Llama-3.2-3B-Instruct", "relative_depth": 0.6},
    "Mistral-7B-Instruct-v0.3": {"hf_name": "mistralai/Mistral-7B-Instruct-v0.3", "relative_depth": 0.6},
    "Qwen2.5-3B-Instruct": {"hf_name": "Qwen/Qwen2.5-3B-Instruct", "relative_depth": 0.6},
    "Qwen2.5-7B-Instruct": {"hf_name": "Qwen/Qwen2.5-7B-Instruct", "relative_depth": 0.6},
    "Phi-3.5-mini-instruct": {"hf_name": "microsoft/Phi-3.5-mini-instruct", "relative_depth": 0.6},
    "Gemma-2-2B": {"hf_name": "google/gemma-2-2b", "relative_depth": 0.6},
    "Gemma-2-27B": {"hf_name": "google/gemma-2-27b", "relative_depth": 0.6},
}


def _log(msg: str):
    print(msg, flush=True)


def _batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def save_model_tensor(records, vectors, dataset_name, model_label, out_dir: Path) -> Path:
    hidden = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).reshape(len(records), 2, -1)
    payload = {
        "hidden": hidden,
        "texts": [r.texts for r in records],
        "anchor_ids": [r.anchor_id for r in records],
        "meta": {
            "dataset": dataset_name,
            "model_label": model_label,
            "set_size": 2,
            "languages": ["en", "fr"],
        },
    }
    slug = sanitize_model_name(model_label)
    path = out_dir / f"{dataset_name}__{slug}.pt"
    torch.save(payload, path)
    return path


def extract_for_model(records, model_label, batch_size, out_dir: Path):
    spec = MODEL_SPECS[model_label]
    extractor = ActivationExtractor(
        spec["hf_name"],
        relative_depth=spec["relative_depth"],
        max_length=256,
    )
    vectors = []
    prompts = [text for r in records for text in r.texts]
    for batch_idx, batch in enumerate(_batched(prompts, batch_size), start=1):
        _log(f"[extract:{model_label}] batch {batch_idx} ({len(batch)} texts)")
        vec = extractor.extract_batch(batch)
        vectors.extend(vec)

    out_path = save_model_tensor(records, vectors, records[0].dataset, model_label, out_dir)
    del extractor.model
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


def build_dataset(dataset_name: str, n_anchors: int, model_labels: list[str], batch_size: int = 8):
    out_dir = BENCH_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[dataset] building records for {dataset_name} with n_anchors={n_anchors}")
    records = build_records(dataset_name, n_anchors)
    save_record_manifest(records, out_dir)
    save_benchmark_manifest(records, out_dir)

    status = {
        "dataset": dataset_name,
        "n_anchors": len(records),
        "models": {},
    }
    for model_label in model_labels:
        try:
            _log(f"[dataset] extracting {dataset_name} for {model_label}")
            out_path = extract_for_model(records, model_label, batch_size=batch_size, out_dir=out_dir)
            status["models"][model_label] = {"status": "ok", "path": str(out_path)}
        except Exception as exc:
            status["models"][model_label] = {"status": "deferred", "reason": repr(exc)}
            _log(f"[dataset] deferred {model_label}: {exc!r}")

    status_path = out_dir / "extraction_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status_path


if __name__ == "__main__":
    default_models = [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Phi-3.5-mini-instruct",
        "Gemma-2-2B",
    ]

    _log("Building multilingual benchmark datasets...")
    build_dataset("wmt14_fr_en", n_anchors=2048, model_labels=default_models, batch_size=8)
    build_dataset("opus100_multi_en", n_anchors=4096, model_labels=default_models, batch_size=8)
    _log("Done.")

