from typing import Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from transformers.cache_utils import StaticCache

from . import TreeProcessor, CandidateExtras, InferenceExtras, TrainingExtras
from ..util import get_mask_mod_w_offset


class FixedTreeProcessor(TreeProcessor):
    def __init__(
        self,
        paths: Sequence[Sequence[int]],
        ignore_threshold: float | None,
        mask_token_id: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        # The idea is the following:
        # For a block at each position we create a divergent tree of the shape as defined by paths
        # - Path (0,0,1) means 'top-0', 'top-0', 'top-1' sampling
        # Ignore threshold means, 'don't' have this token if the target assigns this token a p < ignore_threshold


        distances_from_left_most = []
        left_most_ancestors = []
        for path in paths:
            i = 0
            while (
                i < len(path)
                and i < len(left_most_path)
                and path[i] == left_most_path[i]
            ):
                i += 1
            distances_from_left_most.append(len(path) - i)
            left_most_ancestors.append(i)
        new_idx = sorted(
            range(len(paths)),
            key=lambda i: (
                distances_from_left_most[i] * len(left_most_path) + len(paths[i])
            ),
        )
        new_idx_reverse = sorted(range(len(paths)), key=lambda i: new_idx[i])
        paths = [[new_idx_reverse[j] for j in paths[i]] for i in new_idx]
        top_k = [top_k[i] for i in new_idx]
        self.distance_to_left_most = torch.tensor(
            [distances_from_left_most[i] for i in new_idx],
            dtype=torch.long,
            device=device,
        )
        self.left_most_ancestors = torch.tensor(
            [left_most_ancestors[i] for i in new_idx], dtype=torch.long, device=device
        )
        self.is_left_most = self.distance_to_left_most == 0

        self.paths = [tuple(path) for path in paths]
        self.top_k = torch.tensor(list(top_k), dtype=torch.long, device=device)
        self.MASK_TOKEN_ID = mask_token_id

        self.tree_size = len(self.paths)
        self.parent = torch.tensor(
            [path[-2] if len(path) > 1 else -1 for path in self.paths],
            dtype=torch.long,
            device=device,
        )
        self.full_tree_mask = torch.zeros(
            (self.tree_size, self.tree_size), dtype=torch.bool, device=device
        )
        self.is_leaf = torch.ones((self.tree_size,), dtype=torch.bool, device=device)
        self.seq_positions = torch.zeros(
            (self.tree_size,), dtype=torch.long, device=device
        )

        for vertex_idx, path in enumerate(self.paths):
            ancestor_path = path[:-1]
            self.full_tree_mask[vertex_idx, vertex_idx] = True
            self.seq_positions[vertex_idx] = len(ancestor_path)
            if len(ancestor_path) > 0:
                parent_idx = ancestor_path[-1]
                self.parent[vertex_idx] = parent_idx
                self.is_leaf[parent_idx] = False
                for ancestor_idx in ancestor_path:
                    self.full_tree_mask[vertex_idx, ancestor_idx] = True

        self.leaf_indices = torch.nonzero(self.is_leaf, as_tuple=True)[0]
        self.attn_tree_mask = self.full_tree_mask[~self.is_leaf & ~self.is_left_most][
            :, ~self.is_leaf & ~self.is_left_most
        ]
        self.no_leaf_no_left_mask = self.attn_tree_mask

    def construct_training_extras(
        self, input_ids, anchors, document_mask, position_ids, target
    ):
        B, S = input_ids.shape
        B, N_T = anchors.shape
        target_hidden_states, tree_labels = self._generate_labels(
            input_ids=input_ids,
            position_ids=position_ids,
            document_masks=document_mask,
            anchors=anchors,
            target=target,
        )  # target_hidden_states: [B, S, D], tree_labels: [B, N_T, T]

        noise_input_ids = torch.gather(input_ids, 1, anchors)
        noise_input_ids = F.pad(
            noise_input_ids[:, :, None],
            (0, self.tree_size - 1),
            value=self.MASK_TOKEN_ID,
        )
        noise_embds = target.get_input_embeddings()(noise_input_ids)

        sequence_position_ids = self.seq_positions[None, None, :] + anchors[:, :, None]
        tree_position_ids = torch.arange(self.tree_size, device=input_ids.device)[
            None, None, :
        ].expand(B, N_T, -1)

        return TrainingExtras(
            tree_labels=tree_labels,
            noise_embds=noise_embds,
            sequence_position_ids=sequence_position_ids,
            tree_position_ids=tree_position_ids,
            target_hidden_states=target_hidden_states,
            tree_masks=self.full_tree_mask[None, None, :, :].expand(B, N_T, -1, -1),
        )

    def construct_candidate_extras(self, drafted_ids, sequence_position_ids):
        return CandidateExtras(
            input_ids=drafted_ids[:, 0],
            sequence_position_ids=sequence_position_ids[:, 0],
            tree_masks=self.full_tree_mask[None, :],
            parents_idx=self.parent[None, :],
        )

    def construct_inference_extras(self, input_ids, target):
        return InferenceExtras(
            tree_masks=self.full_tree_mask[None, None, :, :].expand(1, 1, -1, -1),
            sequence_position_ids=self.seq_positions[None, None, :].expand(1, 1, -1)
            + input_ids.shape[1] - 1,
            noise_embds=target.get_input_embeddings()(
                torch.tensor(
                    [[input_ids[0, -1], *[self.MASK_TOKEN_ID] * (self.tree_size - 1)]],
                    device=input_ids.device,
                )
            ).view(1, 1, self.tree_size, -1),
            tree_position_ids=torch.arange(self.tree_size, device=input_ids.device),
        )

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

        TS_MOD = self.no_leaf_no_left_mask.shape[0]

        past_key_values = StaticCache(
            config=target.config,
            max_cache_len=S + N_B * TS_MOD,
        )
        prefill_cache_position = torch.arange(S, device=input_ids.device)

        prefill_out = target(
            input_ids=input_ids,
            position_ids=position_ids,
            # output_hidden_states=True,
            attn_mask=prefill_mask,
            past_key_values=past_key_values,
            cache_position=prefill_cache_position,
            use_cache=True,
        )
        target_hidden_states = prefill_out.hidden_states
        logits = prefill_out.logits

        left_most_indexes = torch.nonzero(self.is_left_most, as_tuple=True)[0]
        tree_labels = torch.zeros(
            (B, N_B, self.tree_size), dtype=torch.long, device=input_ids.device
        )
        for i in left_most_indexes:
            tree_labels[:, :, i] = torch.gather(input_ids, 1, anchors + i)
            is_child = self.parent == i
            this_logits = torch.gather(
                logits, 1, i + anchors[:, :, None].expand(-1, -1, logits.shape[-1])
            )  # [B, N_B, vocab_size]
            this_logits.scatter_(-1, tree_labels[:, :, i : i + 1], -torch.inf)
            this_logits_topk = torch.topk(this_logits, k=8, dim=-1).indices
            child_ranks = self.top_k[is_child].long()
            tree_labels[:, :, is_child] = this_logits_topk[:, :, child_ranks]

        next_input_idx = torch.arange(self.tree_size, device=input_ids.device)[
            (self.distance_to_left_most == 1) & ~self.is_leaf
        ]
        TS_MOD = self.no_leaf_no_left_mask.shape[0]

        def mask_mod(B, _H, Q, KV):
            q_block = Q % N_B
            k_block = (KV - S) % N_B
            q_tree_pos = Q // N_B % TS_MOD
            k_tree_pos = (KV - S) // N_B % TS_MOD
            is_ctxt = KV < S
            is_causal = KV < anchors[B, q_block] + self.left_most_ancestors[q_tree_pos]
            same_doc = True
            if document_masks is None:
                same_doc = True
            else:
                same_doc = (
                    document_masks[B, KV % S] == document_masks[B, anchors[B, q_block]]
                )
            ctxt_part = is_ctxt & is_causal & same_doc
            is_ancestor = self.no_leaf_no_left_mask[q_tree_pos, k_tree_pos]
            tree_part = ~is_ctxt & (q_block == k_block) & is_ancestor
            return ctxt_part | tree_part

        tree_gen_block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=None,
            Q_LEN=N_B * TS_MOD,
            KV_LEN=S + N_B * TS_MOD,
            device=input_ids.device,
        )

        curr_cache_pos = S
        while next_input_idx.size(0) > 0:
            position_ids = (
                (
                    self.seq_positions[next_input_idx][None, None, :]
                    + anchors[:, :, None]
                )
                .transpose(1, 2)
                .reshape(B, -1)
            )  # [B, width * N_NB]
            input_ids = (
                tree_labels[:, :, next_input_idx].transpose(1, 2).reshape(B, -1)
            )  # [B, width * N_B]
            start_idx = next_input_idx.min().item()
            end_idx = next_input_idx.max().item() + 1
            block_index = (start_idx * N_B) // tree_gen_block_mask.BLOCK_SIZE[0]
            block_end = (end_idx * N_B) // tree_gen_block_mask.BLOCK_SIZE[0]
            this_mask = tree_gen_block_mask[:, :, block_index:block_end]
            this_mask.mask_mod = get_mask_mod_w_offset(
                this_mask.mask_mod, _offset=start_idx * N_B
            )
            outputs = target(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=past_key_values,
                cache_position=(
                    curr_cache_pos + torch.arange(input_ids.shape[1], device=input_ids.device)
                ),
                attn_mask=this_mask,
            )
            curr_cache_pos += input_ids.shape[1]
            this_logits = outputs.logits.view(B, -1, N_B, logits.shape[-1]).transpose(
                1, 2
            )  # [B, N_B, width, vocab_size]
            this_logits_topk = torch.topk(this_logits, k=8, dim=-1).indices
            all_children = torch.zeros(
                (self.tree_size), dtype=torch.bool, device=input_ids.device
            )
            # is_any_child = (self.parent[None, :] == next_input_idx[:, None]).any(dim=0)
            for i, par in enumerate(next_input_idx):
                children = self.parent == par
                tree_labels[:, :, children] = this_logits_topk[
                    :, :, i, self.top_k[children]
                ]
                all_children[children] = True
            next_input_idx = torch.nonzero(all_children & ~self.is_leaf, as_tuple=True)[
                0
            ]
            
        return target_hidden_states, tree_labels
