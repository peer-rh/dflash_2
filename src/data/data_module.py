from dataclasses import dataclass, field
import math
import os
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
from .eval_data import load_and_process_dataset


@dataclass
class DataModuleConfig:
    data_path: str = ""
    batch_size: int = 64

    seq_len: int = 3072
    n_blocks: int = 64
    block_size: int = 16
    precomputed_tree_path: str | None = None

    num_workers: int = 4
    seed: int = 42

    n_validation_samples: int = 2048

    quality_datasets: list[str] = field(
        default_factory=lambda: ["gsm8k", "alpaca", "humaneval"]
    )
    n_samples_per_quality_dataset: int = 16


def _pack_token_sequences(
    prompt_sequences: list[list[int]],
    response_sequences: list[list[int]],
    *,
    seq_len: int,
    block_size: int,
    pad_token_id: int,
    sequence_indices: list[int] | None = None,
) -> dict[str, list[Any]]:
    flex_bs = 128
    total_lengths = [len(prompt) + len(response) for prompt, response in zip(prompt_sequences, response_sequences)]
    padded_seq_lens = [math.ceil(length / flex_bs) * flex_bs for length in total_lengths]
    sorted_indices = sorted(
        range(len(total_lengths)),
        key=lambda idx: padded_seq_lens[idx],
        reverse=True,
    )

    bucket_lengths: list[int] = []
    bucket_seq_counts: list[int] = []
    bucket_input_ids: list[torch.Tensor] = []
    bucket_masks: list[torch.Tensor] = []
    bucket_position_ids: list[torch.Tensor] = []
    bucket_answer_intervals: list[list[list[int]]] = []
    bucket_response_sequence_idx: list[torch.Tensor] | None = [] if sequence_indices is not None else None
    bucket_response_row_idx: list[torch.Tensor] | None = [] if sequence_indices is not None else None

    for idx in sorted_indices:
        prompt_ids = prompt_sequences[idx]
        response_ids = response_sequences[idx]
        prompt_len = len(prompt_ids)
        response_len = len(response_ids)
        total_len = total_lengths[idx]
        if total_len > seq_len or response_len < block_size + 1:
            continue

        try:
            bucket_idx = next(
                j
                for j, used_len in enumerate(bucket_lengths)
                if used_len + padded_seq_lens[idx] <= seq_len
            )
        except StopIteration:
            bucket_idx = len(bucket_lengths)
            bucket_input_ids.append(torch.full((seq_len,), pad_token_id, dtype=torch.long))
            bucket_masks.append(torch.zeros((seq_len,), dtype=torch.long))
            bucket_position_ids.append(torch.zeros((seq_len,), dtype=torch.long))
            bucket_answer_intervals.append([])
            bucket_lengths.append(0)
            bucket_seq_counts.append(0)
            if bucket_response_sequence_idx is not None and bucket_response_row_idx is not None:
                bucket_response_sequence_idx.append(torch.full((seq_len,), -1, dtype=torch.long))
                bucket_response_row_idx.append(torch.full((seq_len,), -1, dtype=torch.long))

        seq_start = bucket_lengths[bucket_idx]
        seq_end = seq_start + total_len
        response_start = seq_start + prompt_len

        bucket_input_ids[bucket_idx][seq_start:seq_end] = torch.tensor(
            prompt_ids + response_ids, dtype=torch.long
        )
        bucket_masks[bucket_idx][seq_start:seq_end] = bucket_seq_counts[bucket_idx] + 1
        bucket_position_ids[bucket_idx][seq_start:seq_end] = torch.arange(total_len, dtype=torch.long)
        bucket_answer_intervals[bucket_idx].append([response_start, seq_end])

        if (
            sequence_indices is not None
            and bucket_response_sequence_idx is not None
            and bucket_response_row_idx is not None
        ):
            bucket_response_sequence_idx[bucket_idx][response_start:seq_end] = sequence_indices[idx]
            bucket_response_row_idx[bucket_idx][response_start:seq_end] = torch.arange(
                response_len, dtype=torch.long
            )

        bucket_seq_counts[bucket_idx] += 1
        bucket_lengths[bucket_idx] += padded_seq_lens[idx]

    packed = {
        "input_ids": [tensor.tolist() for tensor in bucket_input_ids],
        "masks": [tensor.tolist() for tensor in bucket_masks],
        "position_ids": [tensor.tolist() for tensor in bucket_position_ids],
        "answer_intervals": bucket_answer_intervals,
    }
    if bucket_response_sequence_idx is not None and bucket_response_row_idx is not None:
        packed["response_sequence_idx"] = [tensor.tolist() for tensor in bucket_response_sequence_idx]
        packed["response_row_idx"] = [tensor.tolist() for tensor in bucket_response_row_idx]
    return packed


def setup_synthetic_dataset(
    tokenizer, data_path, seq_len, block_size, num_workers, seed, n_validation
):
    datafiles = []
    if data_path.startswith("."):
        for i in os.listdir(data_path):
            if i.endswith(".jsonl"):
                datafiles.append(os.path.join(data_path, i))
        dataset = load_dataset("json", data_files=datafiles)
        dataset = dataset["train"]
    else:
        dataset = load_dataset(data_path, split="train")

    print(f"Loaded synthetic dataset with {len(dataset)} samples. Processing...")

    def preprocess(batch):
        prompts = tokenizer(batch["prompt"], add_special_tokens=False)["input_ids"]
        responses = tokenizer(batch["response"], add_special_tokens=False)["input_ids"]
        return _pack_token_sequences(
            prompts,
            responses,
            seq_len=seq_len,
            block_size=block_size,
            pad_token_id=tokenizer.pad_token_id,
        )

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



def setup_precomputed_tree_dataset(
    precomputed_tree_path: str,
    pad_token_id: int,
    seq_len: int,
    block_size: int,
    seed: int,
    n_validation: int,
) -> tuple[Dataset, Dataset]:
    with h5py.File(precomputed_tree_path, "r") as hf:
        prompt_ids = [np.asarray(ids, dtype=np.int64).tolist() for ids in hf["prompt_ids"]]
        response_ids = [np.asarray(ids, dtype=np.int64).tolist() for ids in hf["response_ids"]]
        sequence_offsets = np.asarray(hf["sequence_offsets"][:], dtype=np.int64)
        sub_trees = hf["sub_trees"]
        sub_trees_ar_probs = hf["sub_trees_ar_probs"]

        if sequence_offsets.shape[0] != len(prompt_ids) + 1:
            raise ValueError(
                "Invalid precomputed tree HDF5: `sequence_offsets` must have exactly "
                "len(prompt_ids) + 1 entries."
            )
        expected_rows = int(sequence_offsets[-1])
        if sub_trees.shape[0] != expected_rows or sub_trees_ar_probs.shape[0] != expected_rows:
            raise ValueError(
                "Invalid precomputed tree HDF5: row counts in `sub_trees` / "
                "`sub_trees_ar_probs` do not match `sequence_offsets[-1]`."
            )

    packed = _pack_token_sequences(
        prompt_ids,
        response_ids,
        seq_len=seq_len,
        block_size=block_size,
        pad_token_id=pad_token_id,
        sequence_indices=list(range(len(prompt_ids))),
    )
    dataset = Dataset.from_dict(packed)
    dataset = dataset.shuffle(seed=seed)
    print(
        f"Loaded precomputed tree dataset with {len(prompt_ids)} sequences and "
        f"{len(dataset)} packed samples."
    )
    val_dataset = dataset.select(range(min(n_validation, len(dataset))))
    train_dataset = dataset.select(range(min(n_validation, len(dataset)), len(dataset)))
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
        if self.config.precomputed_tree_path is not None:
            self.train_dataset, self.val_dataset = setup_precomputed_tree_dataset(
                self.config.precomputed_tree_path,
                self.pad_token_id,
                self.config.seq_len,
                self.config.block_size,
                self.config.seed,
                self.config.n_validation_samples,
            )
        else:
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
        response_sequence_idx = None
        response_row_idx = None
        if "response_sequence_idx" in batch[0]:
            response_sequence_idx = torch.stack(
                [torch.tensor(x["response_sequence_idx"], dtype=torch.long) for x in batch]
            )
            response_row_idx = torch.stack(
                [torch.tensor(x["response_row_idx"], dtype=torch.long) for x in batch]
            )

        for i, b in enumerate(batch):
            for interval in b["answer_intervals"]:
                potential_anchor_positions[
                    i, interval[0] : interval[1] - 1 - self.config.block_size
                ] = 1
        anchors = torch.multinomial(
            potential_anchor_positions, self.config.n_blocks, replacement=False
        )
        anchors = torch.sort(anchors, dim=1).values
        batch_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "document_mask": masks,
            "anchors": anchors,
        }
        if response_sequence_idx is not None and response_row_idx is not None:
            batch_dict["anchor_sequence_idx"] = torch.gather(response_sequence_idx, 1, anchors)
            batch_dict["anchor_response_idx"] = torch.gather(response_row_idx, 1, anchors)
        return batch_dict

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
