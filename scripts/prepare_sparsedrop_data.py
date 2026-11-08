#!/usr/bin/env python3
"""Pin and pack WikiText-103; each nonempty source row ends in EOS.

Produces flat uint16 token arrays; the trainer uses seq_len consecutive inputs
and their next-token labels with stride seq_len, without crossing split boundaries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tokenizer', default='HuggingFaceTB/SmolLM2-135M')
    parser.add_argument('--dataset', default='Salesforce/wikitext')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    tok_revision = api.model_info(args.tokenizer).sha
    data_revision = api.dataset_info(args.dataset).sha
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=tok_revision)
    if len(tokenizer) > 65536 or tokenizer.eos_token_id is None:
        raise ValueError('uint16 packing requires vocab <=65536 and an EOS token')
    tokenizer.save_pretrained(out / 'tokenizer')
    dataset = load_dataset(args.dataset, 'wikitext-103-v1', revision=data_revision)
    metadata = dict(dataset=args.dataset, dataset_config='wikitext-103-v1',
                    dataset_revision=data_revision, tokenizer=args.tokenizer,
                    tokenizer_revision=tok_revision, vocab_size=len(tokenizer),
                    eos_token_id=tokenizer.eos_token_id, dtype='uint16',
                    packing='nonblank rows in source order + EOS; flat vector; split-disjoint',
                    splits={})
    for split in ('train', 'validation', 'test'):
        raw = out / f'.{split}.raw.tmp'
        count = 0
        with raw.open('wb') as stream:
            for offset in range(0, len(dataset[split]), 1000):
                rows = [s for s in dataset[split][offset:offset+1000]['text'] if s.strip()]
                encoded = tokenizer(rows, add_special_tokens=False, return_attention_mask=False)['input_ids']
                for ids in encoded:
                    ids.append(tokenizer.eos_token_id)
                    np.asarray(ids, dtype=np.uint16).tofile(stream)
                    count += len(ids)
                if offset % 100000 == 0:
                    print(json.dumps({'split': split, 'rows': offset, 'tokens': count}), flush=True)
        packed = np.lib.format.open_memmap(out / f'.{split}.npy.tmp', mode='w+', dtype=np.uint16, shape=(count,))
        source = np.memmap(raw, mode='r', dtype=np.uint16)
        packed[:] = source[:]
        packed.flush()
        del packed, source
        os.replace(out / f'.{split}.npy.tmp', out / f'{split}.npy')
        raw.unlink()
        digest = hashlib.sha256()
        with (out / f'{split}.npy').open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        metadata['splits'][split] = {'tokens': count, 'rows': len(dataset[split]), 'sha256': digest.hexdigest()}
        print(json.dumps(metadata['splits'][split]), flush=True)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
