from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

import src.data.data_module as data_module_module
from src.data.data_module import (
    DataModule,
    DataModuleConfig,
    _pack_token_sequences,
    setup_precomputed_tree_dataset,
)
from src.trainer import Trainer
from src.trees import TrainingExtras, TreeInfo
from src.trees.block_tree import BlockTree
from src.trees.every_branch_tree import EveryBranchTreeProcessor
from src.trees.fixed_tree_prunable import PrunableTreeProcessor


def _create_precomputed_tree_h5(path: Path, tree_size: int = 16) -> tuple[list[list[int]], list[list[int]]]:
    prompt_ids = [
        [11, 12],
        [31, 32, 33],
    ]
    response_ids = [
        [21, 22, 23, 24, 25, 26],
        [41, 42, 43, 44, 45, 46],
    ]
    total_rows = sum(len(response) for response in response_ids)
    tree_rows = np.zeros((total_rows, tree_size), dtype=np.int64)
    prob_rows = np.zeros((total_rows, tree_size), dtype=np.float32)

    offsets = [0]
    row_id = 0
    for seq_idx, response in enumerate(response_ids):
        for resp_idx, token in enumerate(response):
            tree_rows[row_id] = np.arange(tree_size, dtype=np.int64) + row_id * 100
            tree_rows[row_id, 0] = token
            prob_rows[row_id] = np.linspace(0.05, 0.95, tree_size, dtype=np.float32)
            prob_rows[row_id, 0] = 0.25 + 0.1 * seq_idx + 0.01 * resp_idx
            row_id += 1
        offsets.append(row_id)

    with h5py.File(path, "w") as hf:
        vlen_int64 = h5py.vlen_dtype(np.int64)
        hf.create_dataset("prompt_ids", data=np.array([np.asarray(x, dtype=np.int64) for x in prompt_ids], dtype=object), dtype=vlen_int64)
        hf.create_dataset("response_ids", data=np.array([np.asarray(x, dtype=np.int64) for x in response_ids], dtype=object), dtype=vlen_int64)
        hf.create_dataset("sub_trees", data=tree_rows)
        hf.create_dataset("sub_trees_ar_probs", data=prob_rows)
        hf.create_dataset("sequence_offsets", data=np.asarray(offsets, dtype=np.int64))

    return prompt_ids, response_ids


def _pack_precomputed_tree_h5_eager(
    path: Path,
    *,
    pad_token_id: int,
    seq_len: int,
    block_size: int,
) -> list[dict[str, list[int] | list[list[int]]]]:
    with h5py.File(path, "r") as hf:
        prompt_ids = [np.asarray(ids, dtype=np.int64).tolist() for ids in hf["prompt_ids"]]
        response_ids = [np.asarray(ids, dtype=np.int64).tolist() for ids in hf["response_ids"]]

    packed = _pack_token_sequences(
        prompt_ids,
        response_ids,
        seq_len=seq_len,
        block_size=block_size,
        pad_token_id=pad_token_id,
        sequence_indices=list(range(len(prompt_ids))),
    )

    samples = []
    for idx in range(len(packed["input_ids"])):
        sample = {
            "input_ids": packed["input_ids"][idx],
            "masks": packed["masks"][idx],
            "position_ids": packed["position_ids"][idx],
            "answer_intervals": packed["answer_intervals"][idx],
        }
        if "response_sequence_idx" in packed:
            sample["response_sequence_idx"] = packed["response_sequence_idx"][idx]
        if "response_row_idx" in packed:
            sample["response_row_idx"] = packed["response_row_idx"][idx]
        samples.append(sample)
    return samples


def _normalize_packed_sample(sample: dict[str, list[int] | list[list[int]]]) -> tuple:
    def normalize(value):
        if isinstance(value, list):
            return tuple(normalize(item) for item in value)
        return value

    return tuple((key, normalize(sample[key])) for key in sorted(sample))


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0


class _FakeTarget:
    def __init__(self, vocab_size: int = 256, hidden_size: int = 8):
        self.config = SimpleNamespace()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embedding


def _bind_trainer_methods(dummy_trainer):
    for name in [
        "_compute_sibling_overlap_loss",
        "_get_anchor_chunk_size",
        "_get_ce_chunk_size",
        "_slice_tree_info",
        "_slice_training_extras",
        "_build_drafter_attention_mask",
        "_prepare_batch",
        "_compute_loss_and_metrics",
        "_process_prepared_batch",
        "process_batch",
    ]:
        setattr(dummy_trainer, name, getattr(Trainer, name).__get__(dummy_trainer, Trainer))
    return dummy_trainer


class _DummyChunkingTreeProcessor:
    def __init__(self, n_trees: int = 2, tree_size: int = 2, hidden_size: int = 8, supports_chunking: bool = True):
        self.n_trees = n_trees
        self.tree_size = tree_size
        self.hidden_size = hidden_size
        self.supports_chunking_flag = supports_chunking
        self.construct_training_extras_call_count = 0
        self.anchor_sequence_idx = None
        self.anchor_response_idx = None
        tree_mask = torch.tensor([[True, False], [True, True]])
        self.tree_info = TreeInfo(
            tree_mask=tree_mask.view(1, 1, tree_size, tree_size).expand(1, n_trees, -1, -1),
            parent_idx=torch.tensor([-1, 0]).view(1, 1, tree_size).expand(1, n_trees, -1),
            depth=torch.tensor([0, 1]).view(1, 1, tree_size).expand(1, n_trees, -1),
            is_leaf=torch.tensor([False, True]).view(1, 1, tree_size).expand(1, n_trees, -1),
            relation_map=torch.zeros(1, n_trees, tree_size, tree_size, dtype=torch.long),
            tree_position_ids=torch.arange(tree_size).view(1, 1, tree_size).expand(1, n_trees, -1),
        )

    def supports_anchor_chunking(self) -> bool:
        return self.supports_chunking_flag

    def get_parent_idx(self):
        return torch.tensor([-1, 0])

    def construct_training_extras(
        self,
        input_ids,
        anchors,
        document_mask,
        position_ids,
        target,
        anchor_sequence_idx=None,
        anchor_response_idx=None,
    ):
        self.construct_training_extras_call_count += 1
        self.anchor_sequence_idx = anchor_sequence_idx
        self.anchor_response_idx = anchor_response_idx

        batch_size, n_trees = anchors.shape
        child_labels = anchors.remainder(self.hidden_size - 1) + 1
        tree_labels = torch.stack([torch.zeros_like(child_labels), child_labels], dim=-1)
        noise_embds = torch.zeros(batch_size, n_trees, self.tree_size, self.hidden_size)
        noise_embds.scatter_(
            3,
            child_labels[:, :, None, None].expand(-1, -1, 1, 1),
            5.0,
        )
        sequence_position_ids = anchors[:, :, None] + torch.arange(self.tree_size).view(1, 1, self.tree_size)
        return TrainingExtras(
            tree_labels=tree_labels,
            seq_labels=tree_labels,
            tree_ar_prob=torch.ones(batch_size, n_trees, self.tree_size),
            tree_cum_prob=torch.ones(batch_size, n_trees, self.tree_size),
            noise_embds=noise_embds,
            sequence_position_ids=sequence_position_ids,
            target_hidden_states=[torch.zeros(batch_size, input_ids.shape[1], self.hidden_size)],
            tree_info=TreeInfo(
                tree_mask=self.tree_info.tree_mask.expand(batch_size, n_trees, -1, -1),
                parent_idx=self.tree_info.parent_idx.expand(batch_size, n_trees, -1),
                depth=self.tree_info.depth.expand(batch_size, n_trees, -1),
                is_leaf=self.tree_info.is_leaf.expand(batch_size, n_trees, -1),
                relation_map=self.tree_info.relation_map.expand(batch_size, n_trees, -1, -1),
                tree_position_ids=self.tree_info.tree_position_ids.expand(batch_size, n_trees, -1),
            ),
        )


class _CountingDrafter:
    def __init__(self, q_head=None):
        self.config = SimpleNamespace(use_q_head=q_head is not None)
        self.forward_call_count = 0
        self.q_head = q_head

    def extract_ctx_features(self, hidden_states):
        return hidden_states[0]

    def __call__(self, hidden_states, target_ctx_features, attention_mask, position_ids, tree_info):
        self.forward_call_count += 1
        return hidden_states, hidden_states


def _make_dummy_trainer(*, anchor_chunk_size=None, n_trees: int = 2, supports_chunking: bool = True):
    hidden_size = 8
    tree_processor = _DummyChunkingTreeProcessor(
        n_trees=n_trees,
        tree_size=2,
        hidden_size=hidden_size,
        supports_chunking=supports_chunking,
    )
    dummy_trainer = SimpleNamespace(
        config=SimpleNamespace(
            verbose=False,
            loss_weighting=None,
            sibling_overlap_loss_enabled=False,
            sibling_overlap_loss_weight=0.0,
            sibling_overlap_temperature=0.5,
            sibling_overlap_topk=8,
            anchor_chunk_size=anchor_chunk_size,
            ce_chunk_size=None,
        ),
        tree_processor=tree_processor,
        target=object(),
        drafter=_CountingDrafter(),
        lm_head=lambda x: x,
        tokenizer=None,
        sibling_pair_i=torch.tensor([], dtype=torch.long),
        sibling_pair_j=torch.tensor([], dtype=torch.long),
        num_sibling_pairs=0,
    )
    return _bind_trainer_methods(dummy_trainer)


class _StaticTreeProcessor:
    def __init__(self, training_extras: TrainingExtras, parent_idx: torch.Tensor):
        self.training_extras = training_extras
        self.parent_idx = parent_idx

    def supports_anchor_chunking(self) -> bool:
        return True

    def get_parent_idx(self) -> torch.Tensor:
        return self.parent_idx

    def construct_training_extras(
        self,
        input_ids,
        anchors,
        document_mask,
        position_ids,
        target,
        anchor_sequence_idx=None,
        anchor_response_idx=None,
    ):
        return self.training_extras


def _make_tree_info(parent_idx: torch.Tensor, tree_mask: torch.Tensor, batch_size: int, n_trees: int) -> TreeInfo:
    tree_size = parent_idx.numel()
    depth = torch.zeros(tree_size, dtype=torch.long)
    for node_idx in range(tree_size):
        curr_idx = int(node_idx)
        while parent_idx[curr_idx] >= 0:
            depth[node_idx] += 1
            curr_idx = int(parent_idx[curr_idx].item())

    is_leaf = torch.ones(tree_size, dtype=torch.bool)
    is_leaf[parent_idx[parent_idx >= 0]] = False

    return TreeInfo(
        tree_mask=tree_mask.view(1, 1, tree_size, tree_size).expand(batch_size, n_trees, -1, -1),
        parent_idx=parent_idx.view(1, 1, tree_size).expand(batch_size, n_trees, -1),
        depth=depth.view(1, 1, tree_size).expand(batch_size, n_trees, -1),
        is_leaf=is_leaf.view(1, 1, tree_size).expand(batch_size, n_trees, -1),
        relation_map=torch.zeros(batch_size, n_trees, tree_size, tree_size, dtype=torch.long),
        tree_position_ids=torch.arange(tree_size).view(1, 1, tree_size).expand(batch_size, n_trees, -1),
    )


def _make_static_trainer(
    training_extras: TrainingExtras,
    parent_idx: torch.Tensor,
    *,
    ce_chunk_size=None,
    config_overrides=None,
    q_head=None,
):
    config_overrides = config_overrides or {}
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            verbose=False,
            loss_weighting=None,
            sibling_overlap_loss_enabled=False,
            sibling_overlap_loss_weight=0.0,
            sibling_overlap_temperature=0.5,
            sibling_overlap_topk=8,
            anchor_chunk_size=None,
            ce_chunk_size=ce_chunk_size,
            **config_overrides,
        ),
        tree_processor=_StaticTreeProcessor(training_extras, parent_idx),
        target=object(),
        drafter=_CountingDrafter(q_head=q_head),
        lm_head=lambda x: x,
        tokenizer=None,
    )
    pred_parent_idx = parent_idx[1:]
    same_parent = pred_parent_idx[:, None] == pred_parent_idx[None, :]
    valid_parent = (pred_parent_idx[:, None] >= 0) & (pred_parent_idx[None, :] >= 0)
    distinct_nodes = ~torch.eye(pred_parent_idx.shape[0], dtype=torch.bool)
    upper_triangle = torch.triu(torch.ones_like(same_parent, dtype=torch.bool), diagonal=1)
    sibling_pairs = (same_parent & valid_parent & distinct_nodes & upper_triangle).nonzero(as_tuple=False)
    trainer.sibling_pair_i = sibling_pairs[:, 0]
    trainer.sibling_pair_j = sibling_pairs[:, 1]
    trainer.num_sibling_pairs = int(sibling_pairs.shape[0])
    return _bind_trainer_methods(trainer)


def _assert_loss_metrics_match(loss_a, metrics_a, loss_b, metrics_b, metric_keys):
    assert torch.allclose(loss_a, loss_b, atol=1e-6, rtol=0.0)
    for key in metric_keys:
        value_a = metrics_a[key]
        value_b = metrics_b[key]
        if isinstance(value_a, torch.Tensor):
            assert torch.allclose(value_a, value_b, atol=1e-6, rtol=0.0), key
        else:
            assert value_a == value_b, key


def _make_dummy_batch(anchors: torch.Tensor) -> dict[str, torch.Tensor]:
    input_ids = torch.zeros((1, 6), dtype=torch.long)
    for anchor in anchors[0]:
        input_ids[0, anchor + 1] = (anchor % 7) + 1
    return {
        "input_ids": input_ids,
        "anchors": anchors,
        "document_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
        "anchor_sequence_idx": torch.arange(anchors.shape[1]).view(1, -1),
        "anchor_response_idx": torch.arange(anchors.shape[1]).view(1, -1) + 10,
    }


def test_precomputed_tree_dataset_emits_anchor_lookup_metadata(tmp_path, monkeypatch):
    h5_path = tmp_path / "trees.h5"
    _, response_ids = _create_precomputed_tree_h5(h5_path)
    monkeypatch.setattr("src.data.data_module.AutoTokenizer.from_pretrained", lambda _: _DummyTokenizer())

    config = DataModuleConfig(
        precomputed_tree_path=str(h5_path),
        batch_size=1,
        seq_len=32,
        n_blocks=2,
        block_size=2,
        n_validation_samples=0,
        quality_datasets=[],
    )
    data_module = DataModule(config, target="dummy")
    train_dataset, _ = setup_precomputed_tree_dataset(
        str(h5_path),
        data_module.pad_token_id,
        config.seq_len,
        config.block_size,
        config.seed,
        config.n_validation_samples,
    )

    torch.manual_seed(0)
    batch = data_module._collate_batch([train_dataset[0]])
    assert "anchor_sequence_idx" in batch
    assert "anchor_response_idx" in batch
    assert batch["anchor_sequence_idx"].shape == (1, config.n_blocks)
    assert batch["anchor_response_idx"].shape == (1, config.n_blocks)

    anchor_tokens = batch["input_ids"].gather(1, batch["anchors"])[0].tolist()
    for token, seq_idx, resp_idx in zip(
        anchor_tokens,
        batch["anchor_sequence_idx"][0].tolist(),
        batch["anchor_response_idx"][0].tolist(),
    ):
        assert seq_idx >= 0
        assert resp_idx >= 0
        assert token == response_ids[seq_idx][resp_idx]


def test_precomputed_tree_dataset_chunked_matches_eager_reference(tmp_path):
    h5_path = tmp_path / "trees.h5"
    _create_precomputed_tree_h5(h5_path)

    expected_samples = _pack_precomputed_tree_h5_eager(
        h5_path,
        pad_token_id=0,
        seq_len=32,
        block_size=2,
    )
    expected_normalized = sorted(_normalize_packed_sample(sample) for sample in expected_samples)

    datasets = []
    for chunk_size in (1, 2):
        dataset, _ = setup_precomputed_tree_dataset(
            str(h5_path),
            0,
            32,
            2,
            seed=42,
            n_validation=0,
            precomputed_tree_read_chunk_size=chunk_size,
        )
        datasets.append(dataset)

    for dataset in datasets:
        actual_normalized = sorted(
            _normalize_packed_sample(dataset[idx]) for idx in range(len(dataset))
        )
        assert actual_normalized == expected_normalized


def test_data_module_preprocess_supports_chunked_h5_with_loader_workers(tmp_path, monkeypatch):
    h5_path = tmp_path / "trees.h5"
    _create_precomputed_tree_h5(h5_path)
    monkeypatch.setattr("src.data.data_module.AutoTokenizer.from_pretrained", lambda _: _DummyTokenizer())
    original_from_generator = data_module_module.Dataset.from_generator
    captured_kwargs: dict[str, object] = {}

    def recording_from_generator(cls, generator, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return original_from_generator(generator, *args, **kwargs)

    monkeypatch.setattr(
        data_module_module.Dataset,
        "from_generator",
        classmethod(recording_from_generator),
    )

    config = DataModuleConfig(
        precomputed_tree_path=str(h5_path),
        precomputed_tree_read_chunk_size=1,
        batch_size=1,
        seq_len=32,
        n_blocks=2,
        block_size=2,
        num_workers=3,
        n_validation_samples=0,
        quality_datasets=[],
    )
    data_module = DataModule(config, target="dummy")
    data_module.preprocess()

    assert len(data_module.train_dataset) > 0
    sample = data_module.train_dataset[0]
    assert "num_proc" not in captured_kwargs
    assert "response_sequence_idx" in sample
    assert "response_row_idx" in sample


def test_every_branch_offline_labels_load_from_h5(tmp_path):
    h5_path = tmp_path / "trees.h5"
    _create_precomputed_tree_h5(h5_path)
    processor = EveryBranchTreeProcessor(
        depth=2,
        n_candidate_tokens=None,
        n_compute_branches=4,
        mask_token_id=0,
        device=torch.device("cpu"),
        labels_h5_path=str(h5_path),
    )
    fake_target = _FakeTarget()

    dummy_hidden_states = [torch.ones(1, 16, 8)]
    processor._prefill_target = lambda **_: (dummy_hidden_states, None, None)  # type: ignore[method-assign]

    input_ids = torch.tensor([[11, 12, 21, 22, 23, 24, 25, 26, 0, 0, 0, 0, 0, 0, 0, 0]])
    anchors = torch.tensor([[2, 4]])
    document_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    anchor_sequence_idx = torch.tensor([[0, 0]])
    anchor_response_idx = torch.tensor([[0, 2]])

    extras = processor.construct_training_extras(
        input_ids,
        anchors,
        document_mask,
        position_ids,
        fake_target,
        anchor_sequence_idx=anchor_sequence_idx,
        anchor_response_idx=anchor_response_idx,
    )

    assert torch.equal(extras.tree_labels[:, :, 0], input_ids.gather(1, anchors))
    assert torch.all(extras.tree_ar_prob[:, :, 0] == 1.0)
    expected_cum_prob = torch.where(
        processor.full_tree_mask,
        extras.tree_ar_prob[:, :, None, :],
        1.0,
    ).prod(dim=-1)
    assert torch.allclose(extras.tree_cum_prob, expected_cum_prob)
    assert extras.target_hidden_states is dummy_hidden_states


def test_every_branch_offline_labels_validate_tree_width(tmp_path):
    bad_h5 = tmp_path / "bad.h5"
    with h5py.File(bad_h5, "w") as hf:
        vlen_int64 = h5py.vlen_dtype(np.int64)
        hf.create_dataset("prompt_ids", data=np.array([np.asarray([1, 2], dtype=np.int64)], dtype=object), dtype=vlen_int64)
        hf.create_dataset("response_ids", data=np.array([np.asarray([3, 4, 5], dtype=np.int64)], dtype=object), dtype=vlen_int64)
        hf.create_dataset("sub_trees", data=np.zeros((3, 15), dtype=np.int64))
        hf.create_dataset("sub_trees_ar_probs", data=np.zeros((3, 15), dtype=np.float32))
        hf.create_dataset("sequence_offsets", data=np.asarray([0, 3], dtype=np.int64))

    processor = EveryBranchTreeProcessor(
        depth=2,
        n_candidate_tokens=None,
        n_compute_branches=4,
        mask_token_id=0,
        device=torch.device("cpu"),
        labels_h5_path=str(bad_h5),
    )
    try:
        processor._ensure_labels_h5()
    except ValueError as exc:
        assert "tree size" in str(exc)
    else:
        raise AssertionError("Expected width validation to fail for mismatched HDF5 tree size.")


def test_trainer_process_batch_passes_offline_anchor_metadata():
    batch_size = 1
    n_trees = 2
    tree_size = 2
    vocab_size = 8
    hidden_size = 4

    tree_mask = torch.tensor([[True, False], [True, True]])
    tree_info = TreeInfo(
        tree_mask=tree_mask.view(1, 1, tree_size, tree_size).expand(batch_size, n_trees, -1, -1),
        parent_idx=torch.tensor([-1, 0]).view(1, 1, tree_size).expand(batch_size, n_trees, -1),
        depth=torch.tensor([0, 1]).view(1, 1, tree_size).expand(batch_size, n_trees, -1),
        is_leaf=torch.tensor([False, True]).view(1, 1, tree_size).expand(batch_size, n_trees, -1),
        relation_map=torch.zeros(batch_size, n_trees, tree_size, tree_size, dtype=torch.long),
        tree_position_ids=torch.arange(tree_size).view(1, 1, tree_size).expand(batch_size, n_trees, -1),
    )

    class RecordingTreeProcessor:
        def __init__(self):
            self.anchor_sequence_idx = None
            self.anchor_response_idx = None

        def construct_training_extras(
            self,
            input_ids,
            anchors,
            document_mask,
            position_ids,
            target,
            anchor_sequence_idx=None,
            anchor_response_idx=None,
        ):
            self.anchor_sequence_idx = anchor_sequence_idx
            self.anchor_response_idx = anchor_response_idx
            noise_embds = torch.zeros(batch_size, n_trees, tree_size, hidden_size)
            sequence_position_ids = anchors[:, :, None] + torch.arange(tree_size).view(1, 1, tree_size)
            return TrainingExtras(
                tree_labels=torch.tensor([[[1, 2], [3, 4]]]),
                seq_labels=torch.tensor([[[1, 2], [3, 4]]]),
                tree_ar_prob=torch.ones(batch_size, n_trees, tree_size),
                tree_cum_prob=torch.ones(batch_size, n_trees, tree_size),
                noise_embds=noise_embds,
                sequence_position_ids=sequence_position_ids,
                target_hidden_states=[torch.zeros(batch_size, 6, hidden_size)],
                tree_info=tree_info,
            )

    class DummyDrafter:
        def __init__(self):
            self.config = SimpleNamespace(use_q_head=False)

        def extract_ctx_features(self, hidden_states):
            return hidden_states[0]

        def __call__(self, hidden_states, target_ctx_features, attention_mask, position_ids, tree_info):
            batch, steps, hidden = hidden_states.shape
            outputs = torch.zeros(batch, steps, hidden)
            return outputs, outputs

    dummy_trainer = SimpleNamespace(
        config=SimpleNamespace(
            verbose=False,
            loss_weighting=None,
            sibling_overlap_loss_enabled=False,
            sibling_overlap_loss_weight=0.0,
            sibling_overlap_temperature=0.5,
            sibling_overlap_topk=8,
            anchor_chunk_size=None,
            ce_chunk_size=None,
        ),
        tree_processor=RecordingTreeProcessor(),
        target=object(),
        drafter=DummyDrafter(),
        lm_head=lambda x: torch.zeros(*x.shape[:-1], vocab_size),
        tokenizer=None,
        sibling_pair_i=torch.tensor([], dtype=torch.long),
        sibling_pair_j=torch.tensor([], dtype=torch.long),
        num_sibling_pairs=0,
    )
    dummy_trainer = _bind_trainer_methods(dummy_trainer)

    batch = {
        "input_ids": torch.tensor([[5, 6, 7, 8, 9, 0]]),
        "anchors": torch.tensor([[1, 2]]),
        "document_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
        "position_ids": torch.arange(6).unsqueeze(0),
        "anchor_sequence_idx": torch.tensor([[10, 11]]),
        "anchor_response_idx": torch.tensor([[2, 3]]),
    }

    loss, metrics = dummy_trainer.process_batch(batch)
    assert torch.isfinite(loss)
    assert metrics["block_count"] == batch_size * n_trees
    assert torch.equal(dummy_trainer.tree_processor.anchor_sequence_idx, batch["anchor_sequence_idx"])
    assert torch.equal(dummy_trainer.tree_processor.anchor_response_idx, batch["anchor_response_idx"])


def test_every_branch_online_mode_still_uses_generate_labels():
    processor = EveryBranchTreeProcessor(
        depth=2,
        n_candidate_tokens=None,
        n_compute_branches=4,
        mask_token_id=0,
        device=torch.device("cpu"),
    )
    fake_target = _FakeTarget()

    called = {"value": False}
    dummy_hidden_states = [torch.zeros(1, 6, 8)]
    dummy_labels = torch.arange(processor.tree_size, dtype=torch.long).view(1, 1, -1)
    dummy_probs = torch.ones(1, 1, processor.tree_size)

    def _fake_generate_labels(**kwargs):
        called["value"] = True
        return dummy_hidden_states, dummy_labels, dummy_probs, dummy_probs

    processor._generate_labels = _fake_generate_labels  # type: ignore[method-assign]
    extras = processor.construct_training_extras(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
        anchors=torch.tensor([[2]]),
        document_mask=torch.tensor([[1, 1, 1, 1, 1, 1]]),
        position_ids=torch.arange(6).unsqueeze(0),
        target=fake_target,
    )

    assert called["value"] is True
    assert extras.target_hidden_states is dummy_hidden_states
    assert torch.equal(extras.tree_labels, dummy_labels)


def test_ce_chunking_matches_unchunked_loss_metrics():
    parent_idx = torch.tensor([-1, 0])
    tree_mask = torch.tensor([[True, False], [True, True]])
    anchors = torch.tensor([[1, 3]])
    input_ids = torch.tensor([[0, 0, 2, 0, 4, 0]])
    noise_embds = torch.tensor(
        [[
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 2.5, 0.3, -0.4, -0.8]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1, 0.5, 2.0, -1.0]],
        ]]
    )
    tree_labels = torch.tensor([[[0, 2], [0, 4]]])
    tree_info = _make_tree_info(parent_idx, tree_mask, batch_size=1, n_trees=2)
    training_extras = TrainingExtras(
        tree_labels=tree_labels,
        seq_labels=tree_labels,
        tree_ar_prob=torch.ones_like(tree_labels, dtype=torch.float32),
        tree_cum_prob=torch.ones_like(tree_labels, dtype=torch.float32),
        noise_embds=noise_embds,
        sequence_position_ids=anchors[:, :, None] + torch.arange(2).view(1, 1, 2),
        target_hidden_states=[torch.zeros(1, input_ids.shape[1], noise_embds.shape[-1])],
        tree_info=tree_info,
    )
    batch = {
        "input_ids": input_ids,
        "anchors": anchors,
        "document_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
    }

    trainer_full = _make_static_trainer(training_extras, parent_idx)
    trainer_chunked = _make_static_trainer(training_extras, parent_idx, ce_chunk_size=1)

    loss_full, metrics_full = trainer_full.process_batch(batch)
    loss_chunked, metrics_chunked = trainer_chunked.process_batch(batch)

    _assert_loss_metrics_match(
        loss_full,
        metrics_full,
        loss_chunked,
        metrics_chunked,
        ["lm_loss_sum", "token_correct_count", "accepted_length_sum", "block_count", "token_count"],
    )


def test_ce_chunking_matches_weighted_loss():
    parent_idx = torch.tensor([-1, 0])
    tree_mask = torch.tensor([[True, False], [True, True]])
    anchors = torch.tensor([[1, 3]])
    input_ids = torch.tensor([[0, 0, 2, 0, 4, 0]])
    noise_embds = torch.tensor(
        [[
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 2.2, -0.3, -0.5, -1.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3, 0.4, 1.8, -0.6]],
        ]]
    )
    tree_labels = torch.tensor([[[0, 2], [0, 4]]])
    tree_cum_prob = torch.tensor([[[1.0, 0.2], [1.0, 0.8]]])
    tree_info = _make_tree_info(parent_idx, tree_mask, batch_size=1, n_trees=2)
    training_extras = TrainingExtras(
        tree_labels=tree_labels,
        seq_labels=tree_labels,
        tree_ar_prob=torch.ones_like(tree_labels, dtype=torch.float32),
        tree_cum_prob=tree_cum_prob,
        noise_embds=noise_embds,
        sequence_position_ids=anchors[:, :, None] + torch.arange(2).view(1, 1, 2),
        target_hidden_states=[torch.zeros(1, input_ids.shape[1], noise_embds.shape[-1])],
        tree_info=tree_info,
    )
    batch = {
        "input_ids": input_ids,
        "anchors": anchors,
        "document_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
    }

    config_overrides = {"loss_weighting": "target_probs"}
    trainer_full = _make_static_trainer(training_extras, parent_idx, config_overrides=config_overrides)
    trainer_chunked = _make_static_trainer(
        training_extras,
        parent_idx,
        ce_chunk_size=1,
        config_overrides=config_overrides,
    )

    loss_full, metrics_full = trainer_full.process_batch(batch)
    loss_chunked, metrics_chunked = trainer_chunked.process_batch(batch)

    _assert_loss_metrics_match(
        loss_full,
        metrics_full,
        loss_chunked,
        metrics_chunked,
        ["lm_loss_sum", "token_correct_count", "accepted_length_sum"],
    )


def test_ce_chunking_matches_sibling_overlap():
    parent_idx = torch.tensor([-1, 0, 0])
    tree_mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, False, True],
        ]
    )
    anchors = torch.tensor([[1]])
    input_ids = torch.tensor([[0, 0, 1, 0]])
    noise_embds = torch.tensor(
        [[[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 2.2, 1.0, -0.2, -1.0],
            [0.0, 2.8, 1.8, 1.1, -0.3, -1.2],
        ]]]
    )
    tree_labels = torch.tensor([[[0, 1, 2]]])
    tree_info = _make_tree_info(parent_idx, tree_mask, batch_size=1, n_trees=1)
    training_extras = TrainingExtras(
        tree_labels=tree_labels,
        seq_labels=tree_labels,
        tree_ar_prob=torch.ones_like(tree_labels, dtype=torch.float32),
        tree_cum_prob=torch.tensor([[[1.0, 0.6, 0.4]]]),
        noise_embds=noise_embds,
        sequence_position_ids=torch.tensor([[[1, 2, 2]]]),
        target_hidden_states=[torch.zeros(1, input_ids.shape[1], noise_embds.shape[-1])],
        tree_info=tree_info,
    )
    batch = {
        "input_ids": input_ids,
        "anchors": anchors,
        "document_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
    }

    config_overrides = {
        "sibling_overlap_loss_enabled": True,
        "sibling_overlap_loss_weight": 0.3,
        "sibling_overlap_topk": 3,
        "sibling_overlap_temperature": 0.75,
    }
    trainer_full = _make_static_trainer(training_extras, parent_idx, config_overrides=config_overrides)
    trainer_chunked = _make_static_trainer(
        training_extras,
        parent_idx,
        ce_chunk_size=1,
        config_overrides=config_overrides,
    )

    loss_full, metrics_full = trainer_full.process_batch(batch)
    loss_chunked, metrics_chunked = trainer_chunked.process_batch(batch)

    _assert_loss_metrics_match(
        loss_full,
        metrics_full,
        loss_chunked,
        metrics_chunked,
        [
            "lm_loss_sum",
            "sibling_overlap_loss_sum",
            "sibling_argmax_collision_count",
            "token_correct_count",
            "accepted_length_sum",
        ],
    )


def test_ce_chunking_matches_q_head_metrics():
    parent_idx = torch.tensor([-1, 0])
    tree_mask = torch.tensor([[True, False], [True, True]])
    anchors = torch.tensor([[1, 3]])
    input_ids = torch.tensor([[0, 0, 2, 0, 4, 0]])
    noise_embds = torch.tensor(
        [[
            [[0.0, 0.0, 0.0, 0.0], [0.1, 2.0, 0.3, -0.2]],
            [[0.0, 0.0, 0.0, 0.0], [0.5, -0.1, 1.7, 0.2]],
        ]]
    )
    tree_labels = torch.tensor([[[0, 1], [0, 2]]])
    tree_info = _make_tree_info(parent_idx, tree_mask, batch_size=1, n_trees=2)
    training_extras = TrainingExtras(
        tree_labels=tree_labels,
        seq_labels=tree_labels,
        tree_ar_prob=torch.ones_like(tree_labels, dtype=torch.float32),
        tree_cum_prob=torch.ones_like(tree_labels, dtype=torch.float32),
        noise_embds=noise_embds,
        sequence_position_ids=anchors[:, :, None] + torch.arange(2).view(1, 1, 2),
        target_hidden_states=[torch.zeros(1, input_ids.shape[1], noise_embds.shape[-1])],
        tree_info=tree_info,
    )
    batch = {
        "input_ids": input_ids,
        "anchors": anchors,
        "document_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
    }

    q_head_full = torch.nn.Linear(noise_embds.shape[-1], 1, bias=False)
    with torch.no_grad():
        q_head_full.weight.copy_(torch.tensor([[0.5, -0.25, 0.75, 0.1]]))
    q_head_chunked = torch.nn.Linear(noise_embds.shape[-1], 1, bias=False)
    q_head_chunked.load_state_dict(q_head_full.state_dict())

    trainer_full = _make_static_trainer(training_extras, parent_idx, q_head=q_head_full)
    trainer_chunked = _make_static_trainer(training_extras, parent_idx, ce_chunk_size=1, q_head=q_head_chunked)

    loss_full, metrics_full = trainer_full.process_batch(batch)
    loss_chunked, metrics_chunked = trainer_chunked.process_batch(batch)

    _assert_loss_metrics_match(
        loss_full,
        metrics_full,
        loss_chunked,
        metrics_chunked,
        ["lm_loss_sum", "q_loss_sum", "q_accuracy_count", "token_correct_count"],
    )


def test_ce_chunk_size_validation():
    trainer = _make_dummy_trainer()

    trainer.config.ce_chunk_size = 0
    with pytest.raises(ValueError, match="ce_chunk_size"):
        trainer._get_ce_chunk_size(4)

    trainer.config.ce_chunk_size = -1
    with pytest.raises(ValueError, match="ce_chunk_size"):
        trainer._get_ce_chunk_size(4)
