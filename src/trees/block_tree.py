import torch
from torch.nn.attention.flex_attention import create_block_mask
import torch.nn.functional as F
from . import TreeProcessor, CandidateExtras, InferenceExtras, TrainingExtras, TreeInfo, CHILD_RELATION, DESCENDANT_RELATION, ANCESTOR_RELATION, IS_SELF_RELATION, PARENT_RELATION, expand_tree_info

class BlockTree(TreeProcessor):
    def __init__(self, block_size: int, mask_token_id: int, device: torch.device, random_embds=False):
        super().__init__()
        self.block_size = block_size
        self.MASK_TOKEN_ID = mask_token_id
        self.tree_mask = torch.tril(torch.ones((block_size, block_size), dtype=torch.bool, device=device))
        self.parent_idx = torch.arange(block_size, device=device) - 1
        self.random_embds = random_embds
        depth = torch.arange(block_size, device=device)
        is_leaf = torch.zeros(block_size, dtype=torch.bool, device=device)
        is_leaf[-1] = True
        relation_map = torch.zeros((block_size, block_size), dtype=torch.long, device=device)
        relation_map[depth[None, :] < depth[:, None]] = DESCENDANT_RELATION
        relation_map[depth[None, :] > depth[:, None]] = ANCESTOR_RELATION
        relation_map[depth[None, :] == depth[:, None] + 1] = CHILD_RELATION[0]
        relation_map[depth[None, :] + 1 == depth[:, None] + 1] = PARENT_RELATION
        relation_map[depth[None, :] == depth[:, None]] = IS_SELF_RELATION
        self.tree_info = TreeInfo(
            tree_mask=self.tree_mask,
            parent_idx=self.parent_idx,
            depth=depth,
            is_leaf=is_leaf,
            relation_map=relation_map,
            tree_position_ids=depth

        )

    def construct_candidate_extras(self, drafted_ids, sequence_position_ids, q_values):
        return CandidateExtras(
            input_ids=drafted_ids[:, 0],
            sequence_position_ids=sequence_position_ids[:, 0],
            tree_masks=self.tree_mask[None, :],
            parents_idx=self.parent_idx[None, :]
        )
    
    def construct_inference_extras(self, input_ids, target):
        # input_ids has length curr_pos + 1
        noise_embds = target.get_input_embeddings()(
                torch.tensor(
                    [[input_ids[0, -1], *[self.MASK_TOKEN_ID] * (self.block_size - 1)]],
                    device=input_ids.device,
                )
            ).view(1, 1, self.block_size, -1)
        if self.random_embds:
            torch.nn.init.trunc_normal_(noise_embds[:, :, 1:, :])

        return InferenceExtras(
            tree_info=expand_tree_info(self.tree_info, (1, 1)),
            noise_embds=noise_embds,
            sequence_position_ids=torch.arange(self.block_size, device=input_ids.device)[None, None, :] + input_ids.shape[1] - 1,
        )

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
        B, N_B = anchors.shape
        B, S = input_ids.shape

        def prefill_mask_mod(B, _H, Q, KV):
            is_same_doc = document_mask[B, KV] == document_mask[B, Q]
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

        prefill_out = target(
            input_ids=input_ids,
            position_ids=position_ids,
            # output_hidden_states=True,
            attn_mask=prefill_mask,
        )
        target_hidden_states = prefill_out.hidden_states

        noise_input_ids = torch.gather(input_ids, 1, anchors)
        noise_input_ids = F.pad(
            noise_input_ids[:, :, None],
            (0, self.block_size - 1),
            value=self.MASK_TOKEN_ID,
        )
        noise_embds = target.get_input_embeddings()(noise_input_ids) # [B, N_B, B_S, D]
        if self.random_embds:
            torch.nn.init.trunc_normal_(noise_embds[:, :, 1:, :])
        physical_position_ids = torch.arange(self.block_size, device=input_ids.device)[None, None, :] + anchors[:, :, None]

        tree_labels = torch.gather(input_ids, 1, physical_position_ids.view(B, N_B * self.block_size)).view(B, N_B, self.block_size)

        target_ar_probs = F.softmax(prefill_out.logits[:, :-1], dim=-1) # [B, S-1, V]
        target_ar_probs = torch.gather(target_ar_probs, 2, input_ids[:, 1:, None]).squeeze(2) # [B, S-1] # [:, 0] is pron of token at pos 1
        tree_ar_probs = torch.gather(target_ar_probs, 1, physical_position_ids.view(B, N_B * self.block_size) - 1).view(B, N_B, self.block_size) # [B, N_B, B_S]
        tree_ar_probs[:, :, 0] = 1.0 # We assume collapsed prob at root 
        tree_cum_probs = torch.cumprod(tree_ar_probs, dim=2)

        anchor_position_ids = torch.gather(position_ids, 1, anchors) # [B, N_B]
        sequence_position_ids = torch.arange(self.block_size, device=input_ids.device)[None, None, :] + anchor_position_ids[:, :, None] # [B, N_B, B_S]



        return TrainingExtras(
            tree_labels=tree_labels,
            seq_labels=tree_labels,
            tree_ar_prob=tree_ar_probs,
            tree_cum_prob=tree_cum_probs,
            noise_embds=noise_embds,
            sequence_position_ids=sequence_position_ids,
            tree_info=expand_tree_info(self.tree_info, (B, N_B)),
            target_hidden_states=target_hidden_states,
        )
