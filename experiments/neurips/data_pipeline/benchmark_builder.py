import gzip
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download


DATA_DIR = Path("data")
BENCH_DIR = DATA_DIR / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_model_name(model_name: str) -> str:
    slug = model_name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", slug).strip("_")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


@dataclass
class BenchmarkRecord:
    anchor_id: str
    dataset: str
    split: str
    source_lang: str
    target_lang: str
    texts: list[str]
    meta: dict


def _is_good_pair(en_text: str, fr_text: str, min_chars: int = 20, max_chars: int = 320) -> bool:
    if not en_text or not fr_text:
        return False
    if en_text == fr_text:
        return False
    if len(en_text) < min_chars or len(fr_text) < min_chars:
        return False
    if len(en_text) > max_chars or len(fr_text) > max_chars:
        return False
    return True


def iter_wmt14_fren(limit: int, split: str = "train") -> Iterable[BenchmarkRecord]:
    ds = load_dataset("wmt/wmt14", "fr-en", split=split, streaming=True)
    count = 0
    for idx, row in enumerate(ds):
        tr = row["translation"]
        en_text = normalize_text(tr["en"])
        fr_text = normalize_text(tr["fr"])
        if not _is_good_pair(en_text, fr_text):
            continue
        yield BenchmarkRecord(
            anchor_id=f"wmt14-fr-en-{idx}",
            dataset="wmt14_fr_en",
            split=split,
            source_lang="en",
            target_lang="fr",
            texts=[en_text, fr_text],
            meta={"source": "wmt/wmt14", "config": "fr-en"},
        )
        count += 1
        if count >= limit:
            break


def iter_europarl_fren(limit: int, version: str = "v10") -> Iterable[BenchmarkRecord]:
    relpath = f"data/{version}/training/europarl-{version}.fr-en.tsv.gz"
    local_path = hf_hub_download(repo_id="wmt/europarl", repo_type="dataset", filename=relpath)
    count = 0
    with gzip.open(local_path, "rt", encoding="utf-8", newline="\n") as fh:
        for idx, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            fr_text = normalize_text(parts[0])
            en_text = normalize_text(parts[1])
            if re.fullmatch(r"\d+\.", fr_text) and re.fullmatch(r"\d+\.", en_text):
                continue
            if not _is_good_pair(en_text, fr_text):
                continue
            yield BenchmarkRecord(
                anchor_id=f"europarl-fr-en-{idx}",
                dataset="europarl_fr_en",
                split="train",
                source_lang="en",
                target_lang="fr",
                texts=[en_text, fr_text],
                meta={"source": "wmt/europarl", "version": version, "file": relpath},
            )
            count += 1
            if count >= limit:
                break


def build_records(dataset_name: str, limit: int) -> list[BenchmarkRecord]:
    dataset_name = dataset_name.lower()
    if dataset_name == "wmt14_fr_en":
        return list(iter_wmt14_fren(limit))
    if dataset_name == "europarl_fr_en":
        return list(iter_europarl_fren(limit))
    if dataset_name == "opus100_multi_en":
        # Multilingual follow-up: aggregate multiple English pair directions.
        lang_pairs = [("en", "fr"), ("de", "en"), ("en", "es")]
        per_pair = max(1, limit // len(lang_pairs))
        out: list[BenchmarkRecord] = []
        for src, tgt in lang_pairs:
            cfg = f"{src}-{tgt}"
            ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="train", streaming=True)
            kept = 0
            for idx, row in enumerate(ds):
                tr = row["translation"]
                src_text = normalize_text(tr[src])
                tgt_text = normalize_text(tr[tgt])
                if not _is_good_pair(src_text, tgt_text):
                    continue
                out.append(
                    BenchmarkRecord(
                        anchor_id=f"opus100-{src}-{tgt}-{idx}",
                        dataset="opus100_multi_en",
                        split="train",
                        source_lang=src,
                        target_lang=tgt,
                        texts=[src_text, tgt_text],
                        meta={"source": "Helsinki-NLP/opus-100", "config": cfg},
                    )
                )
                kept += 1
                if kept >= per_pair:
                    break
        return out[:limit]
    raise ValueError(f"Unsupported dataset_name={dataset_name}")


def save_record_manifest(records: list[BenchmarkRecord], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    return path


def save_benchmark_manifest(records: list[BenchmarkRecord], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    payload = {
        "dataset": records[0].dataset if records else None,
        "n_anchors": len(records),
        "set_size": 2,
        "languages": ["en", "fr"],
        "anchor_ids": [r.anchor_id for r in records[:10]],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

