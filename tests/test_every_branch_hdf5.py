from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from src.data.data_module import DataModule, DataModuleConfig, setup_precomputed_tree_dataset
from src.trainer import Trainer
from src.trees import TrainingExtras, TreeInfo
from src.trees.every_branch_tree import EveryBranchTreeProcessor


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


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0


class _FakeTarget:
    def __init__(self, vocab_size: int = 256, hidden_size: int = 8):
        self.config = SimpleNamespace()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embedding


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
        ),
        tree_processor=RecordingTreeProcessor(),
        target=object(),
        drafter=DummyDrafter(),
        lm_head=lambda x: torch.zeros(*x.shape[:-1], vocab_size),
        tokenizer=None,
        sibling_pair_i=torch.tensor([], dtype=torch.long),
        sibling_pair_j=torch.tensor([], dtype=torch.long),
        num_sibling_pairs=0,
        _compute_sibling_overlap_loss=lambda tree_logits: tree_logits.new_zeros(()),
    )

    batch = {
        "input_ids": torch.tensor([[5, 6, 7, 8, 9, 0]]),
        "anchors": torch.tensor([[1, 2]]),
        "document_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
        "position_ids": torch.arange(6).unsqueeze(0),
        "anchor_sequence_idx": torch.tensor([[10, 11]]),
        "anchor_response_idx": torch.tensor([[2, 3]]),
    }

    loss, metrics = Trainer.process_batch(dummy_trainer, batch)
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
