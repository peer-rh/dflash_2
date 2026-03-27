import torch
from typing import Optional
from dataclasses import dataclass

@dataclass
class TreeInfo:
    tree_mask: torch.Tensor # [..., T, T]
    parent_idx: torch.Tensor # [..., T]
    depth: torch.Tensor # [..., T]
    is_leaf: torch.Tensor # [..., T]
    relation_map: torch.Tensor # [..., T, T]
    tree_position_ids: torch.Tensor # [..., T]

UNRELATED_RELATION = 0
PARENT_RELATION = 1
DESCENDANT_RELATION = 2
ANCESTOR_RELATION = 3
IS_SELF_RELATION = 4
CHILD_RELATION = [6, 7, 8, 9]
SIBLING_RELATION = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]



def expand_tree_info(tree_info: TreeInfo, target_shape: tuple[int, ...]) -> TreeInfo:
    return TreeInfo(
        tree_mask=tree_info.tree_mask.expand(*target_shape, -1, -1),
        parent_idx=tree_info.parent_idx.expand(*target_shape, -1),
        depth=tree_info.depth.expand(*target_shape, -1),
        is_leaf=tree_info.is_leaf.expand(*target_shape, -1),
        relation_map=tree_info.relation_map.expand(*target_shape, -1, -1),
        tree_position_ids=tree_info.tree_position_ids.expand(*target_shape, -1),
    )

@dataclass
class TrainingExtras:
    tree_labels: torch.Tensor # [B, N_T, T]
    seq_labels: torch.Tensor # [B, N_T, T']
    tree_ar_prob: torch.Tensor # [B, N_T, T] 
    tree_cum_prob: torch.Tensor # [B, N_T, T] # Note that root token is assumed to have prob 1
    noise_embds: torch.Tensor # [B, N_T, T, D]
    sequence_position_ids: torch.Tensor #[B, N_T, T]
    target_hidden_states: list[torch.Tensor] # tuple([B, S, D],...)
    tree_info: TreeInfo
    

@dataclass
class CandidateExtras:
    input_ids: torch.Tensor # [1, T']
    sequence_position_ids: torch.Tensor # [1, T']
    tree_masks: torch.Tensor # [1, T']
    parents_idx: torch.Tensor # [1, T']

@dataclass
class InferenceExtras:
    noise_embds: torch.Tensor # [1, N_T, T, D]
    sequence_position_ids: torch.Tensor #[1, N_T, T]
    tree_info: TreeInfo

class TreeProcessor:
    parent_idx: torch.Tensor
    def __init__(self):
        pass

    def construct_training_extras(
        self,
        input_ids,
        anchors,
        document_mask,
        position_ids,
        target,
        anchor_sequence_idx: Optional[torch.Tensor] = None,
        anchor_response_idx: Optional[torch.Tensor] = None,
    ) -> TrainingExtras:
        """
        Inputs:
            input_ids: [B, S]
            anchors: [B, N_T]
            document_mask: [B, S]
            position_ids: [B, S]
            target: Model
            anchor_sequence_idx: [B, N_T] optional lookup ids for offline labels
            anchor_response_idx: [B, N_T] optional response-row ids for offline labels
        Returns:
            tree_labels: [B, N_T, T]
            noise_embds: [B, N_T, T, D]
            sequence_position_ids: [B, N_T, T]
            tree_position_ids: [B, N_T, T]
            TODO: target_tree_probs: [B, N_T, T] # Probability of a random walk getting to this node using the verifier_probs
            TODO: tree_masks: [B, N_T, T, T] # Attention mask for each tree
        """
        ...
    
    def construct_inference_extras(self, input_ids, target) -> InferenceExtras:
        """
        Inputs:
            input_ids: [1, S]
            target: Model
        Returns:
            tree_masks: [1, N_T, T, T]
            noise_embds: [1, N_T, T, D]
            sequence_position_ids: [1, N_T, T]
            tree_position_ids: [1, N_T, T]
        """
        ...
    
    def construct_candidate_extras(self, drafted_ids: torch.Tensor, inference_extras: InferenceExtras, q_values: torch.Tensor | None) -> CandidateExtras:
        """
        Inputs:
            drafted_ids: [1, N_T, T]
            inference_extras: InferenceExtras
            q_values: [1, N_T, T]
        Returns:
            input_ids: [1, T']
            sequence_position_ids: [1, T']
            tree_masks: [1, T', T']
            parents_idx: [1, T']
            TODO: probabilities: [1, T'] # For more optimal speculative decoding
        """
        ...

    def get_parent_idx(self):
        """
        Returns: [T]
        """
        return self.parent_idx
