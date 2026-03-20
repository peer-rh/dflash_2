from typing import Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from transformers.cache_utils import StaticCache

from . import DESCENDANT_RELATION, TreeInfo, TreeProcessor, CandidateExtras, InferenceExtras, TrainingExtras, CHILD_RELATION, ANCESTOR_RELATION, IS_SELF_RELATION, PARENT_RELATION, expand_tree_info
from ..util import get_mask_mod_w_offset


class EveryBranchTreeProcessor(TreeProcessor):
    def __init__(
        self,
        depth: int,
        edges: Sequence[tuple[int, int]],
        top_K: Sequence[int],
        n_candidate_tokens: int | None,
        n_compute_branches: int,
        mask_token_id: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.n_candidate_tokens = n_candidate_tokens
        self.n_compute_branches = n_compute_branches
        self.depth = depth
        self.MASK_TOKEN_ID = mask_token_id
        self.top_k = torch.tensor(top_K, device=device)
        # block is [DIST_0, DIST_1, DIST_2, ...]

        self.tree_size = depth * (len(edges) + 1)
        self.full_tree_mask = torch.zeros((self.tree_size, self.tree_size), dtype=torch.bool, device=device)
        self.seq_positions = torch.zeros((self.tree_size,), dtype=torch.long, device=device)
        self.single_branch_mask = torch.zeros((len(edges)+1, len(edges)+1), dtype=torch.bool, device=device)
        self.parent_idx = torch.full((self.tree_size,), -1, dtype=torch.long, device=device)
        self.is_leaf = torch.ones((self.tree_size,), dtype=torch.bool, device=device)


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
        
        self.seq_positions = self.full_tree_mask.sum(dim=-1) - 1
        self.single_dist_to_left_most = self.single_branch_mask.sum(dim=-1) - 1
        self.requires_extra_attention = ~self.is_leaf
        self.requires_extra_attention[:self.depth] = False
        
        relation_map = torch.zeros((self.tree_size, self.tree_size), dtype=torch.long, device=device)
        relation_map[self.full_tree_mask] = ANCESTOR_RELATION
        relation_map[self.full_tree_mask.T] = DESCENDANT_RELATION
        relation_map[self.parent_idx[:, None] == torch.arange(self.tree_size, device=device)[None, :]] = PARENT_RELATION
        relation_map[self.parent_idx[None, :] == torch.arange(self.tree_size, device=device)[:, None]] = CHILD_RELATION
        relation_map[torch.arange(self.tree_size, device=device), torch.arange(self.tree_size, device=device)] = IS_SELF_RELATION
        self.tree_info = TreeInfo(
            tree_mask=self.full_tree_mask,
            parent_idx=self.parent_idx,
            depth=self.seq_positions,
            is_leaf=self.is_leaf,
            relation_map=relation_map,
            tree_position_ids=torch.arange(self.tree_size, device=device)
        )

    def construct_training_extras(
        self, input_ids, anchors, document_mask, position_ids, target
    ):
        B, S = input_ids.shape
        B, N_T = anchors.shape
        target_hidden_states, tree_labels, tree_ar_prob, tree_cum_prob = self._generate_labels(
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

    def construct_candidate_extras(self, drafted_ids: torch.Tensor, inference_extras: InferenceExtras, q_values: torch.Tensor) -> CandidateExtras:
        if self.n_candidate_tokens is None:
            return CandidateExtras(
                input_ids=drafted_ids[:, 0],
                sequence_position_ids=inference_extras.sequence_position_ids[:, 0],
                tree_masks=inference_extras.tree_info.tree_mask[:, 0],
                parents_idx=inference_extras.tree_info.parent_idx[:, 0],
            )
        assert drafted_ids.shape[1] == 1, "Drafted ids should have n_blocks of 1"
        cumulative_prob = torch.where(
            self.full_tree_mask,
            q_values[0, 0, None, :],
            1.0,
        ).prod(dim=-1) + torch.arange(self.tree_size, device=q_values.device) * 1e-6
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

        TS_MOD = self.n_compute_branches * self.requires_extra_attention.sum().item()

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

        tree_labels = torch.zeros(
            (B, N_B, self.tree_size), dtype=torch.long, device=input_ids.device
        )
        tree_ar_prob = torch.zeros(
            (B, N_B, self.tree_size), dtype=logits.dtype, device=input_ids.device
        )
        tree_ar_prob[:, :, 0] = 1.0

        physical_position_ids = torch.arange(self.depth, device=input_ids.device)[None, None, :] + anchors[:, :, None]
        tree_labels[:, :, :self.depth] = torch.gather(input_ids, 1, physical_position_ids.view(B, -1)).view(B, N_B, -1)

        d_0_logits = torch.gather(
            logits, 1, physical_position_ids.view(B, -1).expand(-1, -1, logits.shape[-1])
        ).view(B, N_B, self.depth, -1).softmax(dim=-1) # [B, N_B, depth, vocab_size]
        tree_ar_prob[:, :, 1:self.depth] = torch.gather(
            d_0_logits[:, :, :-1], 3, tree_labels[:, :, 1:self.depth, None]
        ).squeeze(-1).detach() # [B, N_B, depth-1
        d_0_logits[:, :, :-1].scatter_(3, tree_labels[:, :, 1:self.depth, None], -torch.inf) # Should not be picked
        d_0_logits_top_k = torch.topk(d_0_logits[:, :, :-1], k=8, dim=-1) # [B, N_B, depth-1, topk]
        for i in torch.nonzero(self.single_dist_to_left_most == 1, as_tuple=True)[0]:
            k = self.top_k[i].long()
            tree_labels[:, :, i*self.depth:(1+i)*self.depth] = d_0_logits_top_k.indices[:, :, :, k]
            tree_ar_prob[:, :, i*self.depth:(1+i)*self.depth] = d_0_logits_top_k.values[:, :, :, k].detach()
        
        # Compute cum prod for each vertex, which would require attention
        # Only keep the top n_compute_branchea
        compute_blocks= ... # [B, n_compute_branches]
        compute_vertex_idx = ... # [B, n_compute_branches]

        # Now run for each of these and keep the rest as masked

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
            this_probs = F.softmax(this_logits, dim=-1)
            this_probs_topk = torch.topk(this_probs, k=8, dim=-1) # [B, N_B, width, topk]
            all_children = torch.zeros(
                (self.tree_size), dtype=torch.bool, device=input_ids.device
            )
            # is_any_child = (self.parent[None, :] == next_input_idx[:, None]).any(dim=0)
            for i, par in enumerate(next_input_idx):
                children = self.parent_idx == par
                tree_labels[:, :, children] = this_probs_topk.indices[
                    :, :, i, self.top_k[children]
                ]
                tree_ar_prob[:, :, children] = this_probs_topk.values[
                    :, :, i, self.top_k[children]
                ].detach()
                all_children[children] = True
            next_input_idx = torch.nonzero(all_children & ~self.is_leaf, as_tuple=True)[
                0
            ]

        tree_cum_prob = torch.where(
            self.full_tree_mask, tree_ar_prob[:, :, None, :], 1.0
        ).prod(dim=-1) # [B, N_B, T]
        tree_cum_prob = tree_cum_prob.clone().detach()
        return target_hidden_states.detach(), tree_labels.detach(), tree_ar_prob.detach(), tree_cum_prob.detach()
