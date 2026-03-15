from dataclasses import dataclass, field
import math
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from .eval_data import load_and_process_dataset


@dataclass
class DataModuleConfig:
    data_path: str
    batch_size: int = 64

    seq_len: int = 3072
    n_blocks: int = 64
    block_size: int = 16

    num_workers: int = 4
    seed: int = 42

    n_validation_samples: int = 2048

    quality_datasets: list[str] = field(
        default_factory=lambda: ["gsm8k", "alpaca", "humaneval"]
    )
    n_samples_per_quality_dataset: int = 16


def setup_synthetic_dataset(
    tokenizer, data_path, seq_len, block_size, num_workers, seed, n_validation
):
    datafiles = []
    for i in os.listdir(data_path):
        if i.endswith(".jsonl"):
            datafiles.append(os.path.join(data_path, i))
    dataset = load_dataset("json", data_files=datafiles)
    dataset = dataset["train"]

    print(f"Loaded synthetic dataset with {len(dataset)} samples. Processing...")
    FLEX_BS = 128

    def preprocess(batch):
        # 1. Tokenize
        prompts = tokenizer(batch["prompt"], add_special_tokens=False)["input_ids"]
        responses = tokenizer(batch["response"], add_special_tokens=False)["input_ids"]
        prompt_lengths = [len(p) for p in prompts]
        response_lengths = [len(p) for p in responses]
        total_length = [p + r for p, r in zip(prompt_lengths, response_lengths)]

        padded_seq_lens = [math.ceil(t / FLEX_BS) * FLEX_BS for t in total_length]
        idxs = sorted(
            range(len(total_length)),
            key=lambda i: padded_seq_lens[i],
            reverse=True,
        )

        b_lengths, b_n_seq = [], []
        b_input_ids, b_masks, b_position_ids, b_answer_intervals = [], [], [], []

        for i in idxs:
            if total_length[i] > seq_len or response_lengths[i] < block_size + 1:
                continue
            try:
                found_bucket = next(
                    j
                    for j, l in enumerate(b_lengths)
                    if l + padded_seq_lens[i] <= seq_len
                )
            except StopIteration:
                found_bucket = len(b_lengths)
                b_input_ids.append(
                    torch.full((seq_len,), tokenizer.pad_token_id, dtype=torch.long)
                )
                b_masks.append(torch.zeros((seq_len,), dtype=torch.long))
                b_position_ids.append(torch.zeros((seq_len,), dtype=torch.long))
                b_answer_intervals.append([])
                b_lengths.append(0)
                b_n_seq.append(0)

            seq_start = b_lengths[found_bucket]
            seq_end = seq_start + total_length[i]

            b_input_ids[found_bucket][seq_start:seq_end] = torch.tensor(
                prompts[i] + responses[i], dtype=torch.long
            )
            b_masks[found_bucket][seq_start:seq_end] = b_n_seq[found_bucket] + 1
            b_position_ids[found_bucket][seq_start:seq_end] = torch.arange(
                total_length[i], dtype=torch.long
            )

            b_answer_intervals[found_bucket].append(
                [seq_start + prompt_lengths[i], seq_end]
            )

            b_n_seq[found_bucket] += 1
            b_lengths[found_bucket] += padded_seq_lens[i]
        return {
            "input_ids": [t.tolist() for t in b_input_ids],
            "masks": [t.tolist() for t in b_masks],
            "position_ids": [t.tolist() for t in b_position_ids],
            "answer_intervals": b_answer_intervals,  # list[list[[start,end], ...]]
        }

        # 3. Pack sequences

    dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=10_000,
        num_proc=num_workers,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.shuffle(seed=seed)
    print(
        f"Processed synthetic dataset with {len(dataset)} samples after filtering and shuffling."
    )
    val_dataset = dataset.select(range(n_validation))
    train_dataset = dataset.select(range(n_validation, len(dataset)))
    return train_dataset, val_dataset



def setup_quality_dataset(quality_datasets, seed, n_samples_per_quality_dataset) -> None:
    quality_ds = {}
    for dataset_name in quality_datasets:
        dataset = load_and_process_dataset(dataset_name)
        dataset = dataset.shuffle(seed=seed).select(
            range(n_samples_per_quality_dataset)
        )
        quality_ds[dataset_name] = dataset
    return quality_ds

class DataModule:
    def __init__(self, config: DataModuleConfig | dict[str, Any], target):
        self.config = (
            config
            if isinstance(config, DataModuleConfig)
            else DataModuleConfig(**config)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(target)
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id

    def preprocess(self):
        self.train_dataset, self.val_dataset = setup_synthetic_dataset(
            self.tokenizer,
            self.config.data_path,
            self.config.seq_len,
            self.config.block_size,
            self.config.num_workers,
            self.config.seed,
            self.config.n_validation_samples,
        )
        self.quality_ds = setup_quality_dataset(
            self.config.quality_datasets, self.config.seed, self.config.n_samples_per_quality_dataset
        )

    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack(
            [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        )
        position_ids = torch.stack(
            [torch.tensor(x["position_ids"], dtype=torch.long) for x in batch]
        )
        masks = torch.stack([torch.tensor(x["masks"], dtype=torch.long) for x in batch])
        potential_anchor_positions = torch.zeros((len(batch), self.config.seq_len))

        for i, b in enumerate(batch):
            for interval in b["answer_intervals"]:
                potential_anchor_positions[
                    i, interval[0] : interval[1] - 1 - self.config.block_size
                ] = 1
        anchors = torch.multinomial(
            potential_anchor_positions, self.config.n_blocks, replacement=False
        )
        anchors = torch.sort(anchors, dim=1).values
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "document_mask": masks,
            "anchors": anchors,
        }

    def get_train_loader(self) -> DataLoader:
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_batch,
            drop_last=True,
        )
        return dl

    def get_train_dataloader(self) -> DataLoader:
        return self.get_train_loader()

    def get_val_loader(self) -> DataLoader:
        dl = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_batch,
        )
        return dl

    def get_val_dataloader(self) -> DataLoader:
        return self.get_val_loader()

    def get_quality_loaders(self) -> DataLoader:
        dls = {
            name: DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
            )
            for name, ds in self.quality_ds.items()
        }
        return dls

    def get_quality_dataloaders(self) -> dict[str, DataLoader]:
        return self.get_quality_loaders()
