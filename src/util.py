
from transformers import PreTrainedConfig
from transformers.cache_utils import Cache, DynamicLayer
import torch
import torch.nn.functional as F
from typing import Any, Optional
import time

def merge_metrics(A, B):
    if A is None:
        return B
    if B is None:
        return A
    return {
        key: A[key] + B[key] for key in A.keys()
    }

def sample(logits, temperature=1.0):
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)

class SpecializedDynamicCache(Cache):
    # layers: list[SpecializedStaticCacheLayer]
    def __init__( self, config: PreTrainedConfig):
        layers = [SpecializedDynamicCacheLayer() for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers)
    
    def get_seq_end(self):
        return self.layers[0].seq_end

    def mark_tree_update(
        self, trim_point: int, idx_to_keep: torch.Tensor | None
    ):
        for layer in self.layers:
             layer.mark_tree_update(trim_point, idx_to_keep)

class SpecializedDynamicCacheLayer(DynamicLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_end = 0
        self.idx_to_keep = None
    
    def mark_tree_update(
        self, seq_end: int, idx_to_keep: torch.Tensor | None
    ):
        # We will add the idx to keep in the update
        self.seq_end = seq_end
        self.idx_to_keep = idx_to_keep # [T]
    
    def get_seq_length(self) -> int:
        return self.seq_end + (len(self.idx_to_keep) if self.idx_to_keep is not None else 0)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if self.keys is None or self.values is None:
            self.keys = key_states
            self.values = value_states
            self.is_initialized = True
        elif self.idx_to_keep is not None:
            self.keys  = torch.cat((self.keys[:,:,:self.seq_end], self.keys[:, :, self.idx_to_keep], key_states), dim=-2)
            self.values = torch.cat((self.values[:,:,:self.seq_end], self.values[:, :, self.idx_to_keep], value_states), dim=-2)
        else :
            self.keys = torch.cat([self.keys[:, :, :self.seq_end], key_states], dim=-2)
            self.values = torch.cat([self.values[:, :, :self.seq_end], value_states], dim=-2)
        
        self.seq_end = self.keys.shape[2]
        return self.keys, self.values


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids

def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden

def wall_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def get_mask_mod_w_offset(mask_mod, _offset):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + _offset, kv)

    return _mask_mod
