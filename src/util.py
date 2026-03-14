
from transformers import PreTrainedConfig
from transformers.cache_utils import Cache, StaticLayer
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

class SpecializedStaticCache(Cache):
    def __init__(
        self,
        config: PreTrainedConfig,
        max_cache_len: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)
        layers = [SpecializedStaticCacheLayer(max_cache_len=max_cache_len) for _ in range(config.num_hidden_layers)]
        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
    
    def mark_tree_update(
        self, trim_point: int, idx_to_keep: torch.Tensor
    ):
        for layer in self.layers:
             layer.mark_tree_update(trim_point, idx_to_keep)

class SpecializedStaticCacheLayer(StaticLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_end = 0
        self.idx_to_keep = None
    
    def mark_tree_update(
        self, seq_end: int, idx_to_keep: torch.Tensor
    ):
        # We will add the idx to keep in the update
        self.seq_end = seq_end
        self.idx_to_keep = idx_to_keep # [T]

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
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        assert "cache_position" not in cache_kwargs
        if self.idx_to_keep is not None:
            keep_cache_position = torch.arange(len(self.idx_to_keep), device=key_states.device) + self.seq_end
            self._update_internal(keep_cache_position, self.keys[:, :, self.idx_to_keep], self.values[:, :, self.idx_to_keep])
            self.seq_end += len(self.idx_to_keep)

        cache_position = torch.arange(key_states.shape[-2], device=key_states.device) + self.seq_end
        self._update_internal(cache_position, key_states, value_states)
        self.seq_end += key_states.shape[-2]
        return self.keys, self.values

    def _update_internal(self, cache_position, key_states, value_states):
        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states


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
