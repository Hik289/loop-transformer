"""Small offline checks of content-disjoint packing, resume and exact budgets."""
import hashlib
import json

import numpy as np
import pytest

from scripts.prepare_fineweb_sparsedrop import (
    identity_digest, merge_shards, pack_batches, parse_args, read_shard_result,
    split_for_text, sufficient,
)


class FakeTokenizer:
    eos_token_id = 0

    def __len__(self):
        return 64

    def __call__(self, texts, **kwargs):
        assert kwargs == {"add_special_tokens": False, "return_attention_mask": False}
        return {"input_ids": [[ord(char) % 63 + 1 for char in text] for text in texts]}


def documents_by_split():
    found = {}
    for index in range(1000):
        text = f"document {index}"
        found.setdefault(split_for_text(text, 42, 250000, 250000), text)
        if len(found) == 3:
            return found
    raise AssertionError("Fixture did not produce all splits")


def test_content_hash_is_id_independent_and_reproducible():
    text = "same text with different source IDs"
    expected = int.from_bytes(hashlib.sha256(b"fineweb-sparsedrop-v1:42\0" + text.encode()).digest()[:8], "big") % 1000000
    split = "validation" if expected < 1000 else "test" if expected < 2000 else "train"
    assert split_for_text(text) == split_for_text(text) == split
    with pytest.raises(ValueError):
        split_for_text(text, validation_ppm=600000, test_ppm=400000)


def test_packing_and_resume_checksum(tmp_path):
    docs = documents_by_split()
    result = pack_batches([[docs["train"], docs["validation"]], ["", docs["test"], docs["train"]]],
                          FakeTokenizer(), tmp_path, "fixed", {"path": "a.parquet"}, 42, 250000, 250000)
    assert result["rows"] == 5 and result["blank_rows"] == 1
    for split in ("train", "validation", "test"):
        expected = [ord(char) % 63 + 1 for char in docs[split]] + [0]
        if split == "train":
            expected *= 2
        assert np.fromfile(tmp_path / f"{split}.raw", dtype="<u2").tolist() == expected
    assert read_shard_result(tmp_path, "fixed")["splits"] == result["splits"]
    with pytest.raises(ValueError, match="settings changed"):
        read_shard_result(tmp_path, "different")
    path = tmp_path / "train.raw"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        read_shard_result(tmp_path, "fixed")


def test_interrupted_shard_has_no_commit_and_restarts(tmp_path):
    def interrupted():
        yield ["before interruption"]
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        pack_batches(interrupted(), FakeTokenizer(), tmp_path, "fixed", {"path": "a"}, 42, 1000, 1000)
    assert read_shard_result(tmp_path, "fixed") is None
    result = pack_batches([["after restart"]], FakeTokenizer(), tmp_path, "fixed", {"path": "a"}, 42, 1000, 1000)
    assert sum(info["documents"] for info in result["splits"].values()) == 1


def test_exact_prefix_merge_across_shards_and_metadata_last(tmp_path):
    docs = documents_by_split()
    shards = [pack_batches([list(docs.values())], FakeTokenizer(), tmp_path / str(i), "fixed",
                           {"path": f"{i}.parquet"}, 42, 250000, 250000) for i in range(2)]
    targets = {split: shards[0]["splits"][split]["tokens"] + 3 for split in docs}
    assert not sufficient(shards[:1], targets) and sufficient(shards, targets)
    output = tmp_path / "final"
    metadata = merge_shards(shards, output, targets, {"preparation_identity": "fixed"})
    assert metadata["status"] == "complete"
    assert json.loads((output / "metadata.json").read_text()) == metadata
    for split in docs:
        expected = np.concatenate([np.fromfile(tmp_path / str(i) / f"{split}.raw", dtype="<u2") for i in range(2)])[:targets[split]]
        actual = np.load(output / f"{split}.npy", allow_pickle=False)
        assert actual.dtype == np.uint16
        assert np.array_equal(actual, expected)
        assert hashlib.sha256((output / f"{split}.npy").read_bytes()).hexdigest() == metadata["splits"][split]["sha256"]
    with pytest.raises(ValueError, match="Source exhausted"):
        merge_shards(shards[:1], tmp_path / "insufficient", targets, {})
    assert not (tmp_path / "insufficient" / "metadata.json").exists()


def test_default_pins_and_budget(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path)])
    assert args.num_train_tokens == 20_000_000_001
    assert (args.num_train_tokens - 1) // 512 * 512 == 20_000_000_000
    assert args.num_validation_tokens == args.num_test_tokens == 1_048_577
    assert args.dataset_config == "sample-100BT"
    assert args.workers * args.tokenizer_threads == 32
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path), "--dataset-revision", "main"])
    assert identity_digest({"a": 1, "b": 2}) == identity_digest({"b": 2, "a": 1})
