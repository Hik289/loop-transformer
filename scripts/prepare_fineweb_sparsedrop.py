#!/usr/bin/env python3
"""Prepare an immutable, resumable FineWeb-Edu corpus for SparseDrop.

Source Parquet files are processed in bounded waves, then merged in sorted
source order. Splits are SHA256(content)-assigned before tokenization, so the
same text cannot cross splits. The +1 target token permits exact next-token
budgets (20,000,000,001 packed tokens -> 20B prediction tokens at length 512).
No GPU is used. Only completed, verified per-source shards survive a resume;
an interrupted source shard is reprocessed. Final metadata is the commit marker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
from urllib.request import urlopen

import numpy as np


DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SPLITS = ("train", "validation", "test")
PPM = 1_000_000


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def split_for_text(text: str, seed: int = 42, validation_ppm: int = 1000,
                   test_ppm: int = 1000) -> str:
    if min(validation_ppm, test_ppm) < 0 or validation_ppm + test_ppm >= PPM:
        raise ValueError("Split fractions must be nonnegative and leave training data")
    digest = hashlib.sha256(f"fineweb-sparsedrop-v1:{seed}\0".encode() + text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % PPM
    if bucket < validation_ppm:
        return "validation"
    if bucket < validation_ppm + test_ppm:
        return "test"
    return "train"


def read_shard_result(directory: Path, identity: str, verify_hash: bool = True) -> dict | None:
    path = directory / "complete.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("identity") != identity:
        raise ValueError(f"Preparation settings changed for {directory}")
    for split in SPLITS:
        part = directory / f"{split}.raw"
        info = result["splits"][split]
        if not part.is_file() or part.stat().st_size != info["tokens"] * 2:
            raise ValueError(f"Incomplete or corrupt prepared shard: {part}")
        if verify_hash and file_sha256(part) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {part}")
    result["directory"] = str(directory)
    return result


def pack_batches(batches, tokenizer, directory: Path, identity: str, source: dict,
                 split_seed: int, validation_ppm: int, test_ppm: int,
                 total_rows: int | None = None) -> dict:
    """Pack a source shard; publish its completion marker only after durable data."""
    directory.mkdir(parents=True, exist_ok=True)
    counts = {split: 0 for split in SPLITS}
    documents = {split: 0 for split in SPLITS}
    digests = {split: hashlib.sha256() for split in SPLITS}
    paths = {split: directory / f"{split}.raw.partial" for split in SPLITS}
    streams = {split: paths[split].open("wb") for split in SPLITS}
    rows = blank_rows = 0
    start = time.monotonic()
    try:
        for batch_index, texts in enumerate(batches):
            rows += len(texts)
            valid = [text for text in texts if isinstance(text, str) and text.strip()]
            blank_rows += len(texts) - len(valid)
            encoded = tokenizer(valid, add_special_tokens=False, return_attention_mask=False)["input_ids"] if valid else []
            if len(encoded) != len(valid):
                raise ValueError("Tokenizer returned a different document count")
            for text, ids in zip(valid, encoded):
                split = split_for_text(text, split_seed, validation_ppm, test_ppm)
                ids = list(ids) + [tokenizer.eos_token_id]
                if min(ids) < 0 or max(ids) >= len(tokenizer) or max(ids) >= 65536:
                    raise ValueError("Token IDs cannot be safely packed as uint16")
                data = np.asarray(ids, dtype="<u2").tobytes()
                streams[split].write(data)
                digests[split].update(data)
                counts[split] += len(ids)
                documents[split] += 1
            if batch_index % 100 == 0:
                elapsed = max(time.monotonic() - start, 1e-6)
                emit("shard_progress", source=source["path"], rows=rows,
                     tokens=sum(counts.values()), tokens_per_second=sum(counts.values()) / elapsed,
                     elapsed_seconds=elapsed,
                     estimated_remaining_seconds=(elapsed * (total_rows - rows) / rows if total_rows else None))
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        for stream in streams.values():
            stream.close()
    for split in SPLITS:
        os.replace(paths[split], directory / f"{split}.raw")
    result = {"identity": identity, "source": source, "rows": rows, "blank_rows": blank_rows,
              "elapsed_seconds": time.monotonic() - start,
              "splits": {split: {"tokens": counts[split], "documents": documents[split],
                                  "sha256": digests[split].hexdigest()} for split in SPLITS}}
    atomic_json(directory / "complete.json", result)
    result["directory"] = str(directory)
    emit("shard_complete", source=source["path"], tokens=counts, seconds=result["elapsed_seconds"])
    return result


def discover_sources(dataset: str, revision: str, subset: str) -> list[dict]:
    prefix = "sample/" + subset.removeprefix("sample-")
    url = f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/{prefix}?limit=1000"
    entries = []
    while url:
        with urlopen(url, timeout=120) as response:
            entries.extend(json.load(response))
            links = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', links)
        url = match.group(1) if match else None
    sources = []
    for entry in entries:
        if entry.get("type") == "file" and entry["path"].endswith(".parquet"):
            checksum = entry.get("lfs", {}).get("oid")
            if not checksum or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ValueError(f"No pinned source checksum for {entry['path']}")
            sources.append({"path": entry["path"], "bytes": entry["size"], "sha256": checksum})
    if not sources:
        raise ValueError("No Parquet source files found")
    return sorted(sources, key=lambda item: item["path"])


def process_source(task: dict) -> dict:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(task["tokenizer_threads"])
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    directory = Path(task["directory"])
    previous = read_shard_result(directory, task["identity"])
    if previous is not None:
        emit("shard_reused", source=task["source"]["path"])
        return previous
    source = task["source"]
    emit("download_start", source=source["path"], bytes=source["bytes"])
    for attempt in range(1, 4):
        try:
            downloaded = Path(hf_hub_download(repo_id=task["dataset"], repo_type="dataset",
                                             revision=task["dataset_revision"], filename=source["path"],
                                             local_dir=task["download_dir"]))
            if downloaded.stat().st_size != source["bytes"] or file_sha256(downloaded) != source["sha256"]:
                downloaded.unlink()  # This exact task-owned download must be fetched again.
                raise ValueError(f"Downloaded source checksum mismatch: {source['path']}")
            break
        except Exception as error:
            if attempt == 3:
                raise
            emit("download_retry", source=source["path"], attempt=attempt,
                 reason=f"{type(error).__name__}: {error}")
            time.sleep(5 * attempt)
    tokenizer = AutoTokenizer.from_pretrained(task["tokenizer_dir"], local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for practical corpus preparation")
    parquet = pq.ParquetFile(downloaded)
    batches = (batch.column(0).to_pylist() for batch in parquet.iter_batches(
        batch_size=task["batch_size"], columns=["text"], use_threads=False))
    result = pack_batches(batches, tokenizer, directory, task["identity"], source,
                          task["split_seed"], task["validation_ppm"], task["test_ppm"],
                          total_rows=parquet.metadata.num_rows)
    del parquet
    if not task["keep_downloads"]:
        downloaded.unlink()
    return result


def sufficient(shards: list[dict], targets: dict[str, int]) -> bool:
    return all(sum(shard["splits"][split]["tokens"] for shard in shards) >= targets[split] for split in SPLITS)


def merge_shards(shards: list[dict], output: Path, targets: dict[str, int], metadata: dict) -> dict:
    """Merge exact prefixes, retaining each split's independent document order."""
    if not sufficient(shards, targets):
        raise ValueError("Source exhausted before all requested split token targets were reached")
    output.mkdir(parents=True, exist_ok=True)
    details = {}
    for split in SPLITS:
        target = targets[split]
        temporary = output / f".{split}.npy.partial"
        packed = np.lib.format.open_memmap(temporary, mode="w+", dtype="<u2", shape=(target,))
        offset = 0
        used = []
        for shard in shards:
            if offset == target:
                break
            count = min(shard["splits"][split]["tokens"], target - offset)
            if not count:
                continue
            raw = np.memmap(Path(shard["directory"]) / f"{split}.raw", mode="r", dtype="<u2")
            for start in range(0, count, 8_000_000):
                end = min(start + 8_000_000, count)
                packed[offset + start:offset + end] = raw[start:end]
            used.append({"source": shard["source"]["path"], "tokens": count,
                         "source_split_tokens": shard["splits"][split]["tokens"]})
            offset += count
            del raw
        packed.flush()
        del packed
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        checksum = file_sha256(temporary)
        os.replace(temporary, output / f"{split}.npy")
        details[split] = {"tokens": target, "sha256": checksum, "dtype": "uint16",
                          "source_segments": used,
                          "truncation": "Exact token prefix; final document may be truncated"}
        emit("split_published", split=split, tokens=target, sha256=checksum)
    completed = {**metadata, "status": "complete", "splits": details,
                 "source_files": [{"source": shard["source"], "rows": shard["rows"],
                                   "blank_rows": shard["blank_rows"], "splits": shard["splits"]}
                                  for shard in shards]}
    atomic_json(output / "metadata.json", completed)
    return completed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--dataset-config", choices=("sample-10BT", "sample-100BT"), default="sample-100BT")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--num-train-tokens", type=int, default=20_000_000_001)
    parser.add_argument("--num-validation-tokens", type=int, default=1_048_577)
    parser.add_argument("--num-test-tokens", type=int, default=1_048_577)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-ppm", type=int, default=1000)
    parser.add_argument("--test-ppm", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tokenizer-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--keep-downloads", action="store_true")
    parser.add_argument("--keep-prepared-shards", action="store_true")
    args = parser.parse_args(argv)
    for revision in (args.dataset_revision, args.tokenizer_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            parser.error("Dataset and tokenizer revisions must be pinned 40-character commit hashes")
    for key in ("num_train_tokens", "num_validation_tokens", "num_test_tokens", "workers", "tokenizer_threads", "batch_size"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if min(args.validation_ppm, args.test_ppm) <= 0 or args.validation_ppm + args.test_ppm >= PPM:
        parser.error("Validation/test fractions must be positive and leave training data")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another preparation process already owns this output directory") from error
        return prepare(args, output)


def prepare(args, output: Path) -> int:
    # Keep all downloaded/cache files inside the task-owned output directory.
    os.environ["HF_HOME"] = str(output / "hf_cache")
    os.environ["HF_HUB_CACHE"] = str(output / "hf_cache" / "hub")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(args.tokenizer_threads)
    from transformers import AutoTokenizer

    targets = {split: getattr(args, f"num_{split}_tokens") for split in SPLITS}
    settings = {key: getattr(args, key) for key in ("dataset", "dataset_revision", "dataset_config",
                "tokenizer", "tokenizer_revision", "split_seed", "validation_ppm", "test_ppm")}
    settings.update({"targets": targets, "format_version": 1,
                     "split_algorithm": "sha256(UTF8('fineweb-sparsedrop-v1:' + seed + NUL) + UTF8(text)); first 8 bytes big-endian modulo 1000000",
                     "packing": "nonblank complete source documents in lexicographic Parquet path and row order; tokenize without special tokens, append one EOS; exact split token prefixes"})
    identity = identity_digest(settings)
    marker = output / "preparation.json"
    if marker.exists() and json.loads(marker.read_text()).get("identity") != identity:
        raise ValueError("Output directory belongs to different preparation settings; use another directory")
    if not marker.exists() and any(output.glob("*.npy")):
        raise ValueError("Refusing to overwrite an existing corpus without a matching preparation identity")
    final = output / "metadata.json"
    if final.exists():
        metadata = json.loads(final.read_text())
        if metadata.get("preparation_identity") != identity or metadata.get("status") != "complete":
            raise ValueError("Existing corpus metadata does not match this request")
        for split in SPLITS:
            if file_sha256(output / f"{split}.npy") != metadata["splits"][split]["sha256"]:
                raise ValueError(f"Existing final {split} checksum mismatch")
        emit("already_complete", targets=targets)
        return 0
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision,
                                              cache_dir=output / "hf_cache" / "hub", use_fast=True)
    if not tokenizer.is_fast or len(tokenizer) > 65536 or tokenizer.eos_token_id is None:
        raise ValueError("Requires a fast tokenizer, vocabulary <=65536 and an EOS token")
    tokenizer.save_pretrained(output / "tokenizer")
    sources = discover_sources(args.dataset, args.dataset_revision, args.dataset_config)
    state = {"identity": identity, "settings": settings, "sources": sources, "status": "preparing"}
    if marker.exists() and json.loads(marker.read_text()).get("sources") != sources:
        raise ValueError("Pinned source file listing changed")
    atomic_json(marker, state)
    emit("preparation_start", source_files=len(sources), workers=args.workers, tokenizer_threads=args.tokenizer_threads,
         targets=targets, dataset_revision=args.dataset_revision, identity=identity)
    started = time.monotonic()
    shards = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for wave_start in range(0, len(sources), args.workers):
            wave = sources[wave_start:wave_start + args.workers]
            tasks = []
            for index, source in enumerate(wave, wave_start):
                task = {**settings, "identity": identity, "source": source,
                        "directory": str(output / "prepared_shards" / f"{index:05d}"),
                        "download_dir": str(output / "source_downloads"),
                        "tokenizer_dir": str(output / "tokenizer"), "tokenizer_threads": args.tokenizer_threads,
                        "batch_size": args.batch_size, "keep_downloads": args.keep_downloads}
                tasks.append(pool.submit(process_source, task))
            results = [future.result() for future in as_completed(tasks)]
            shards.extend(sorted(results, key=lambda shard: shard["source"]["path"]))
            counts = {split: sum(shard["splits"][split]["tokens"] for shard in shards) for split in SPLITS}
            elapsed = time.monotonic() - started
            emit("wave_complete", processed_files=len(shards), tokens=counts, elapsed_seconds=elapsed,
                 estimated_remaining_seconds=max(0, targets["train"] - counts["train"]) * elapsed / max(1, counts["train"]))
            atomic_json(marker, {**state, "completed_files": len(shards), "tokens": counts})
            if sufficient(shards, targets):
                break
    # Discard any complete tail of the final wave from the corpus identity.
    for end in range(1, len(shards) + 1):
        if sufficient(shards[:end], targets):
            shards = shards[:end]
            break
    metadata = {**settings, "preparation_identity": identity, "vocab_size": len(tokenizer),
                "eos_token_id": tokenizer.eos_token_id, "dtype": "uint16",
                "document_split": "Content hash before tokenization; identical text always belongs to the same split",
                "preparation_seconds_before_merge": time.monotonic() - started}
    merge_shards(shards, output, targets, metadata)
    atomic_json(marker, {**state, "status": "complete", "tokens": targets})
    if not args.keep_prepared_shards:
        shutil.rmtree(output / "prepared_shards")
    emit("preparation_complete", seconds=time.monotonic() - started,
         targets=targets, metadata=str(output / "metadata.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
