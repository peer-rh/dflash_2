import torch
from typing import Optional
from dataclasses import dataclass

@dataclass
class TrainingExtras:
    tree_labels: torch.Tensor # [B, N_T, T]
    noise_embds: torch.Tensor # [B, N_T, T, D]
    sequence_position_ids: torch.Tensor #[B, N_T, T]
    tree_position_ids: Optional[torch.Tensor] # [B, N_T, T]
    target_hidden_states: list[torch.Tensor] # tuple([B, S, D],...)
    tree_masks: torch.Tensor # [B, N_T, T, T] 

@dataclass
class CandidateExtras:
    input_ids: torch.Tensor # [1, T']
    sequence_position_ids: torch.Tensor # [1, T']
    tree_masks: torch.Tensor # [1, T']
    parents_idx: torch.Tensor # [1, T']

@dataclass
class InferenceExtras:
    tree_masks: torch.Tensor # [1, N_T, T]
    noise_embds: torch.Tensor # [1, N_T, T, D]
    sequence_position_ids: torch.Tensor #[1, N_T, T]
    tree_position_ids: Optional[torch.Tensor] # [1, N_T, T]

class TreeProcessor:
    parent_idx: torch.Tensor
    def __init__(self):
        pass

    def construct_training_extras(self, input_ids, anchors, document_mask, position_ids, target) -> TrainingExtras:
        """
        Inputs:
            input_ids: [B, S]
            anchors: [B, N_T]
            document_mask: [B, S]
            position_ids: [B, S]
            target: Model
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