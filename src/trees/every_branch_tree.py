from typing import Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from transformers.cache_utils import StaticCache

from . import DESCENDANT_RELATION, TreeInfo, TreeProcessor, CandidateExtras, InferenceExtras, TrainingExtras, CHILD_RELATION, ANCESTOR_RELATION, IS_SELF_RELATION, PARENT_RELATION, expand_tree_info, UNRELATED_RELATION, SIBLING_RELATION
from ..util import get_mask_mod_w_offset


class EveryBranchTreeProcessor(TreeProcessor):
    def __init__(
        self,
        depth: int,
        n_candidate_tokens: int | None,
        n_compute_branches: int,
        mask_token_id: int,
        device: torch.device,
        labels_h5_path: str | None = None,
    ) -> None:
        super().__init__()
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7)]
        top_K = [0, 1, 2, 3, 0, 1, 0, 1]
        self.n_candidate_tokens = n_candidate_tokens
        self.n_compute_branches = n_compute_branches
        self.depth = depth
        self.MASK_TOKEN_ID = mask_token_id
        self.top_k = torch.tensor(top_K, device=device)
        self.labels_h5_path = labels_h5_path
        self._labels_h5: h5py.File | None = None
        self._labels_offsets: np.ndarray | None = None

        self.tree_size = depth * (len(edges) + 1)
        self.full_tree_mask = torch.zeros((self.tree_size, self.tree_size), dtype=torch.bool, device=device)
        self.seq_positions = torch.zeros((self.tree_size,), dtype=torch.long, device=device)
        self.single_branch_mask = torch.zeros((len(edges)+1, len(edges)+1), dtype=torch.bool, device=device)
        self.parent_idx = torch.full((self.tree_size,), -1, dtype=torch.long, device=device)
        self.is_leaf = torch.ones((self.tree_size,), dtype=torch.bool, device=device)
        self.single_parent_idx = torch.full((len(edges)+1,), -1, dtype=torch.long, device=device)


        self.full_tree_mask[:self.depth, :self.depth] = torch.tril(torch.ones((self.depth, self.depth), dtype=torch.bool, device=device))
        self.single_branch_mask[:, :1] = True
        self.parent_idx[:self.depth] = torch.arange(self.depth, device=device) - 1
        for parent, child in edges:
            self.is_leaf[parent*self.depth : (1+parent)*self.depth] = False
            self.full_tree_mask[child*self.depth : (child+1)*self.depth] = self.full_tree_mask[parent*self.depth : (1+parent)*self.depth]
            self.full_tree_mask[child*self.depth : (1+child)*self.depth, child*self.depth : (1+child)*self.depth] = torch.eye(self.depth, dtype=torch.bool, device=device)
            self.parent_idx[child*self.depth : (1+child)*self.depth] = parent*self.depth + torch.arange(self.depth, device=device)
            self.single_branch_mask[child] = self.single_branch_mask[parent]
            self.single_branch_mask[child, child] = True
            self.single_parent_idx[child] = parent
        
        self.seq_positions = self.full_tree_mask.sum(dim=-1) - 1
        self.single_dist_to_left_most = self.single_branch_mask.sum(dim=-1) - 1
        self.requires_extra_attention = ~self.is_leaf
        self.requires_extra_attention[:self.depth] = False

        child_relations = torch.tensor(CHILD_RELATION, device=device)
        sibling_relations = torch.tensor(SIBLING_RELATION, device=device)
        
        relation_map = torch.full((self.tree_size, self.tree_size), UNRELATED_RELATION, dtype=torch.long, device=device)
        relation_map[self.full_tree_mask] = ANCESTOR_RELATION
        relation_map[self.full_tree_mask.T] = DESCENDANT_RELATION
        relation_map[self.parent_idx[:, None] == torch.arange(self.tree_size, device=device)[None, :]] = PARENT_RELATION
        is_sibling = self.parent_idx[:, None] == self.parent_idx[None, :]
        top_k_aligned = self.top_k[torch.arange(self.tree_size) // self.depth]
        relation_map[is_sibling] = sibling_relations[top_k_aligned[:, None] * 4 + top_k_aligned[None, :]][is_sibling]
        is_child = self.parent_idx[None, :] == torch.arange(self.tree_size, device=device)[:, None]
        relation_map[is_child] = child_relations[None, top_k_aligned].expand(*is_child.shape)[is_child]
        relation_map[torch.arange(self.tree_size, device=device), torch.arange(self.tree_size, device=device)] = IS_SELF_RELATION
        self.tree_info = TreeInfo(
            tree_mask=self.full_tree_mask,
            parent_idx=self.parent_idx,
            depth=self.seq_positions,
            is_leaf=self.is_leaf,
            relation_map=relation_map,
            tree_position_ids=torch.arange(self.tree_size, device=device)
        )

    def supports_anchor_chunking(self) -> bool:
        return self.labels_h5_path is not None

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
        B, S = input_ids.shape
        B, N_T = anchors.shape
        if self.labels_h5_path is None:
            target_hidden_states, tree_labels, tree_ar_prob, tree_cum_prob = self._generate_labels(
                input_ids=input_ids,
                position_ids=position_ids,
                document_masks=document_mask,
                anchors=anchors,
                target=target,
            )
        else:
            if anchor_sequence_idx is None or anchor_response_idx is None:
                raise ValueError(
                    "Offline EveryBranchTreeProcessor requires `anchor_sequence_idx` and "
                    "`anchor_response_idx` in the batch."
                )
            target_hidden_states, _, _ = self._prefill_target(
                input_ids=input_ids,
                position_ids=position_ids,
                document_masks=document_mask,
                target=target,
                max_cache_len=S,
            )
            tree_labels, tree_ar_prob, tree_cum_prob = self._load_offline_labels(
                anchor_sequence_idx=anchor_sequence_idx,
                anchor_response_idx=anchor_response_idx,
                device=input_ids.device,
            )

        noise_input_ids = torch.gather(input_ids, 1, anchors)
        noise_input_ids = F.pad(
            noise_input_ids[:, :, None],
            (0, self.tree_size - 1),
            value=self.MASK_TOKEN_ID,
        )
        noise_embds = target.get_input_embeddings()(noise_input_ids)

        anchor_position_ids = torch.gather(position_ids, 1, anchors) # [B, N_T]
        sequence_position_ids = self.seq_positions[None, None, :] + anchor_position_ids[:, :, None]

        return TrainingExtras(
            tree_labels=tree_labels,
            seq_labels = tree_labels[:, :, :self.depth],
            tree_ar_prob=tree_ar_prob, 
            tree_cum_prob=tree_cum_prob,
            noise_embds=noise_embds,
            sequence_position_ids=sequence_position_ids,
            target_hidden_states=target_hidden_states,
            tree_info=expand_tree_info(self.tree_info, (B, N_T)),
        )

    def construct_candidate_extras(self, drafted_ids: torch.Tensor, inference_extras: InferenceExtras, q_values: torch.Tensor, draft_probs: torch.Tensor | None = None) -> CandidateExtras:
        if self.n_candidate_tokens is None:
            return CandidateExtras(
                input_ids=drafted_ids[:, 0],
                sequence_position_ids=inference_extras.sequence_position_ids[:, 0],
                tree_masks=inference_extras.tree_info.tree_mask[:, 0],
                parents_idx=inference_extras.tree_info.parent_idx[:, 0],
                draft_probs=draft_probs[:, 0] if draft_probs is not None else None,
            )
        assert drafted_ids.shape[1] == 1, "Drafted ids should have n_blocks of 1"
        cumulative_prob = torch.where(
            self.full_tree_mask,
            q_values[0, 0, None, :],
            1.0,
        ).prod(dim=-1) + torch.arange(self.tree_size, 0, -1, device=q_values.device) * 1e-5
        candidate_idxs = cumulative_prob.topk(k=self.n_candidate_tokens, dim=-1).indices
        candidate_idxs = candidate_idxs.sort().values
        # Original parent indices for the selected candidates
        selected_parents = self.parent_idx[candidate_idxs]

        # Build mapping: old tree index -> new compacted index
        old_to_new = torch.full(
            (self.parent_idx.shape[0],),
            -1,
            dtype=candidate_idxs.dtype,
            device=candidate_idxs.device,
        )
        old_to_new[candidate_idxs] = torch.arange(
            candidate_idxs.shape[0],
            device=candidate_idxs.device,
            dtype=candidate_idxs.dtype,
        )

        # Remap parents into the new compact index space
        remapped_parents = old_to_new[selected_parents]
        remapped_parents[selected_parents == -1] = -1
        return CandidateExtras(
            input_ids=drafted_ids[:, 0, candidate_idxs],
            sequence_position_ids=inference_extras.sequence_position_ids[:, 0, candidate_idxs],
            tree_masks=self.full_tree_mask[None, :, candidate_idxs][:, candidate_idxs],
            parents_idx=remapped_parents[None, :],
            draft_probs=draft_probs[:, 0, candidate_idxs] if draft_probs is not None else None,
        )

    def construct_inference_extras(self, input_ids, target):
        return InferenceExtras(
            tree_info=expand_tree_info(self.tree_info, (1, 1)),
            sequence_position_ids=self.seq_positions[None, None, :].expand(1, 1, -1)
            + input_ids.shape[1] - 1,
            noise_embds=target.get_input_embeddings()(
                torch.tensor(
                    [[input_ids[0, -1], *[self.MASK_TOKEN_ID] * (self.tree_size - 1)]],
                    device=input_ids.device,
                )
            ).view(1, 1, self.tree_size, -1),
        )

    def _ensure_labels_h5(self) -> tuple[h5py.File, np.ndarray]:
        if self.labels_h5_path is None:
            raise RuntimeError("Offline labels were requested without `labels_h5_path`.")
        if self._labels_h5 is None:
            self._labels_h5 = h5py.File(self.labels_h5_path, "r")
            self._labels_offsets = np.asarray(self._labels_h5["sequence_offsets"][:], dtype=np.int64)
            if self._labels_h5["sub_trees"].shape[1] != self.tree_size:
                raise ValueError(
                    "Offline tree label width does not match the hardcoded Every Branch "
                    f"tree size: h5={self._labels_h5['sub_trees'].shape[1]} processor={self.tree_size}"
                )
            if self._labels_h5["sub_trees_ar_probs"].shape[1] != self.tree_size:
                raise ValueError(
                    "Offline tree probability width does not match the hardcoded Every "
                    f"Branch tree size: h5={self._labels_h5['sub_trees_ar_probs'].shape[1]} processor={self.tree_size}"
                )
            if self._labels_offsets.shape[0] != self._labels_h5["prompt_ids"].shape[0] + 1:
                raise ValueError(
                    "Invalid offline label file: `sequence_offsets` length must be "
                    "`len(prompt_ids) + 1`."
                )
        assert self._labels_offsets is not None
        return self._labels_h5, self._labels_offsets

    @torch.no_grad()
    def _load_offline_labels(
        self,
        *,
        anchor_sequence_idx: torch.Tensor,
        anchor_response_idx: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels_h5, sequence_offsets = self._ensure_labels_h5()
        seq_idx = anchor_sequence_idx.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        resp_idx = anchor_response_idx.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
        if (seq_idx < 0).any() or (resp_idx < 0).any():
            raise ValueError("Offline Every Branch anchors must point to response tokens.")

        row_ids = sequence_offsets[seq_idx] + resp_idx
        max_rows = labels_h5["sub_trees"].shape[0]
        if (row_ids < 0).any() or (row_ids >= max_rows).any():
            raise IndexError("Offline Every Branch row lookup is out of bounds for the HDF5 label file.")

        tree_labels_np = np.stack([labels_h5["sub_trees"][int(row_id)] for row_id in row_ids], axis=0)
        tree_ar_prob_np = np.stack(
            [labels_h5["sub_trees_ar_probs"][int(row_id)] for row_id in row_ids],
            axis=0,
        )

        batch_shape = anchor_sequence_idx.shape
        tree_labels = torch.from_numpy(tree_labels_np).to(device=device, dtype=torch.long).view(*batch_shape, self.tree_size)
        tree_ar_prob = torch.from_numpy(tree_ar_prob_np).to(device=device, dtype=torch.float32).view(*batch_shape, self.tree_size)
        tree_ar_prob[:, :, 0] = 1.0
        tree_cum_prob = torch.where(
            self.full_tree_mask,
            tree_ar_prob[:, :, None, :],
            1.0,
        ).prod(dim=-1)
        return tree_labels, tree_ar_prob, tree_cum_prob.detach()

    @torch.no_grad()
    @torch.compiler.disable()
    def _prefill_target(
        self,
        *,
        input_ids,
        position_ids,
        document_masks,
        target,
        max_cache_len: int,
    ):
        B, S = input_ids.shape

        def prefill_mask_mod(B, _H, Q, KV):
            is_same_doc = document_masks[B, KV] == document_masks[B, Q]
            is_causal = KV <= Q
            return is_same_doc & is_causal

        prefill_mask = create_block_mask(
            prefill_mask_mod,
            B=B,
            H=None,
            Q_LEN=S,
            KV_LEN=S,
            device=input_ids.device,
        )

        past_key_values = StaticCache(
            config=target.config,
            max_cache_len=max_cache_len,
        )
        prefill_cache_position = torch.arange(S, device=input_ids.device)
        prefill_out = target(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_mask=prefill_mask,
            past_key_values=past_key_values,
            cache_position=prefill_cache_position,
            use_cache=True,
        )
        return prefill_out.hidden_states, prefill_out.logits, past_key_values

    @torch.no_grad()
    @torch.compiler.disable()
    def _generate_labels(
        self,
        input_ids,
        position_ids,
        document_masks,
        anchors,
        target,
    ):
        B, N_B = anchors.shape
        B, S = input_ids.shape
        target_hidden_states, logits, past_key_values = self._prefill_target(
            input_ids=input_ids,
            position_ids=position_ids,
            document_masks=document_masks,
            target=target,
            max_cache_len=S + self.n_compute_branches,
        )

        tree_labels = torch.full(
            (B, N_B, self.tree_size), self.MASK_TOKEN_ID, dtype=torch.long, device=input_ids.device
        )
        tree_ar_prob = torch.ones(
            (B, N_B, self.tree_size), dtype=logits.dtype, device=input_ids.device
        )
        tree_cum_prob = torch.ones(
            (B, N_B, self.tree_size), dtype=logits.dtype, device=input_ids.device
        )
        tree_ar_prob[:, :, 0] = 1.0

        physical_position_ids = torch.arange(self.depth, device=input_ids.device)[None, None, :] + anchors[:, :, None]
        tree_labels[:, :, :self.depth] = torch.gather(input_ids, 1, physical_position_ids.view(B, -1)).view(B, N_B, -1)

        d_0_probs = torch.gather(
            logits, 1, physical_position_ids.view(B, -1, 1).expand(-1, -1, logits.shape[-1])
        ).view(B, N_B, self.depth, -1).softmax(dim=-1) # [B, N_B, depth, vocab_size]
        tree_ar_prob[:, :, 1:self.depth] = torch.gather(
            d_0_probs[:, :, :-1], 3, tree_labels[:, :, 1:self.depth, None]
        ).squeeze(-1).detach() 
        tree_cum_prob[:, :, :self.depth] = torch.where(
            self.full_tree_mask[:self.depth, :self.depth], tree_ar_prob[:, :, None, :self.depth], 1.0
        ).prod(dim=-1) 
        d_0_probs[:, :, :-1].scatter_(3, tree_labels[:, :, 1:self.depth, None], -torch.inf) # Should not be picked
        d_0_probs_top_k = torch.topk(d_0_probs, k=8, dim=-1) # [B, N_B, depth-1, topk]
        for i in torch.nonzero(self.single_dist_to_left_most == 1, as_tuple=True)[0]:
            k = self.top_k[i].long()
            tree_labels[:, :, i*self.depth:(1+i)*self.depth] = d_0_probs_top_k.indices[:, :, :, k]
            tree_ar_prob[:, :, i*self.depth:(1+i)*self.depth] = d_0_probs_top_k.values[:, :, :, k].detach()
            tree_cum_prob[:, :, i*self.depth:(1+i)*self.depth] = tree_ar_prob[:, :, i*self.depth:(1+i)*self.depth] * tree_cum_prob[:, :, :self.depth] 
        
        # Compute cum prod for each vertex, which would require attention
        # Only keep the top n_compute_branches
        candidates = torch.where(
            ~self.is_leaf[None, None, self.depth:],
            tree_cum_prob[:, :, self.depth:],
            0.0
        ).view(B, -1).topk(self.n_compute_branches)
        compute_blocks = candidates.indices // (self.tree_size - self.depth)
        compute_vertex_idx = candidates.indices % (self.tree_size - self.depth) + self.depth

        def mask_mod(B, _H, Q, KV):
            q_block = compute_blocks[B, Q]
            q_vertex_idx = compute_vertex_idx[B, Q]
            q_depth = q_vertex_idx % self.depth
            kv_safe = (KV - S) % self.n_compute_branches
            kv_block = compute_blocks[B, kv_safe]
            kv_vertex_idx = compute_vertex_idx[B, kv_safe]
            is_ctxt = KV < S
            is_causal = (KV <= anchors[B, q_block] + q_depth)
            same_doc = True
            if document_masks is None:
                same_doc = True
            else:
                same_doc = (
                    document_masks[B, KV % S] == document_masks[B, anchors[B, q_block]]
                )
            ctxt_part = is_ctxt & is_causal & same_doc
            return ctxt_part | (~is_ctxt & (q_block == kv_block) & (kv_vertex_idx == q_vertex_idx))

        tree_gen_block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=None,
            Q_LEN=self.n_compute_branches,
            KV_LEN=S + self.n_compute_branches,
            device=input_ids.device,
        )

        anchored_position_ids = torch.gather(position_ids, 1, anchors) # [B, N_B]
        position_ids = self.seq_positions.gather(0, compute_vertex_idx.view(-1)).view(B, -1) + anchored_position_ids.gather(1, compute_blocks)
        input_ids = tree_labels[torch.arange(B, device=input_ids.device)[:, None].expand(B, self.n_compute_branches), compute_blocks, compute_vertex_idx] # [B, n_compute_branches]

        curr_cache_pos = S
        outputs = target(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=(
                curr_cache_pos + torch.arange(input_ids.shape[1], device=input_ids.device)
            ),
            attn_mask=tree_gen_block_mask,
        )
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        top_k = torch.topk(probs, k=8, dim=-1)
        topk_probs_full = torch.ones((B, N_B, self.tree_size, 8), device=input_ids.device, dtype=probs.dtype)
        topk_indices_full = torch.full((B, N_B, self.tree_size, 8), self.MASK_TOKEN_ID, device=input_ids.device, dtype=torch.long)
        topk_probs_full[torch.arange(B)[:, None].expand(B, self.n_compute_branches), compute_blocks, compute_vertex_idx] = top_k.values.detach()
        topk_indices_full[torch.arange(B)[:, None].expand(B, self.n_compute_branches), compute_blocks, compute_vertex_idx] = top_k.indices

        for i in torch.nonzero(self.single_dist_to_left_most == 2, as_tuple=True)[0]:
            k = self.top_k[i].long()
            parents = self.parent_idx[i*self.depth:(1+i)*self.depth] # [depth]
            tree_labels[:, :, i*self.depth:(1+i)*self.depth] = topk_indices_full[:, :, parents, k]
            tree_ar_prob[:, :, i*self.depth:(1+i)*self.depth] = topk_probs_full[:, :, parents, k].detach()
        
        tree_cum_prob = torch.where(
            self.full_tree_mask, tree_ar_prob[:, :, None, :], 1.0
        ).prod(dim=-1) # [B, N_B, T]
        tree_cum_prob = tree_cum_prob.clone().detach()
        return target_hidden_states.detach(), tree_labels.detach(), tree_ar_prob.detach(), tree_cum_prob.detach()
