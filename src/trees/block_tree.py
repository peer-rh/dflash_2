import torch
from torch.nn.attention.flex_attention import create_block_mask
import torch.nn.functional as F
from . import TreeProcessor, CandidateExtras, InferenceExtras, TrainingExtras

class BlockTree(TreeProcessor):
    def __init__(self, block_size: int, mask_token_id: int, device: torch.device):
        super().__init__()
        self.block_size = block_size
        self.MASK_TOKEN_ID = mask_token_id
        self.tree_mask = torch.tril(torch.ones((block_size, block_size), dtype=torch.bool, device=device))
        self.parent_idx = torch.arange(block_size, device=device) - 1

    def construct_candidate_extras(self, drafted_ids, sequence_position_ids):
        return CandidateExtras(
            input_ids=drafted_ids[:, 0],
            sequence_position_ids=sequence_position_ids[:, 0],
            tree_masks=self.tree_mask[None, :],
            parents_idx=self.parent_idx[None, :]
        )
    
    def construct_inference_extras(self, input_ids, target):
        # input_ids has length curr_pos + 1
        return InferenceExtras(
            tree_masks=self.tree_mask[None, None, :],
            tree_position_ids=None,
            noise_embds=target.get_input_embeddings()(
                torch.tensor(
                    [[input_ids[0, -1], *[self.MASK_TOKEN_ID] * (self.block_size - 1)]],
                    device=input_ids.device,
                )
            ).view(1, 1, self.block_size, -1),
            sequence_position_ids=torch.arange(self.block_size, device=input_ids.device)[None, None, :] + input_ids.shape[1] - 1,
        )

    def construct_training_extras(self, input_ids, anchors, document_mask, position_ids, target):
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
            output_hidden_states=True,
            attn_mask=prefill_mask,
        )
        target_hidden_states = prefill_out.hidden_states

        noise_input_ids = torch.gather(input_ids, 1, anchors)
        noise_input_ids = F.pad(
            noise_input_ids[:, :, None],
            (0, self.block_size - 1),
            value=self.MASK_TOKEN_ID,
        )
        noise_embds = target.get_input_embeddings()(noise_input_ids)
        sequence_position_ids = torch.arange(self.block_size, device=input_ids.device)[None, None, :] + anchors[:, :, None]
        tree_position_ids = torch.arange(self.block_size, device=input_ids.device)[
            None, None, :
        ].expand(B, N_B, -1)
        tree_labels = torch.gather(input_ids, 1, sequence_position_ids.view(B, N_B * self.block_size)).view(B, N_B, self.block_size)

        return TrainingExtras(
            tree_labels=tree_labels,
            noise_embds=noise_embds,
            sequence_position_ids=sequence_position_ids,
            tree_position_ids=tree_position_ids,
            target_hidden_states=target_hidden_states,
            tree_masks=self.tree_mask[None, None, :, :].expand(B, N_B, -1, -1),
        )