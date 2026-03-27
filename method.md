# DFlash: Tree-Based Speculative Decoding

## Overview

DFlash is a speculative decoding system for Qwen3 language models. A small **draft model** proposes a tree of candidate token sequences in a single forward pass; the large frozen **target model** (verifier) runs one forward pass over the entire tree with masked attention to verify candidates. The longest accepted path is committed, yielding multiple tokens per target forward pass.

Acceptance uses the rejection sampling criterion from Leviathan et al. (2023): each candidate token is accepted with probability `min(1, p_target(x) / p_draft(x))`, guaranteeing the output distribution matches the target exactly.

---

## Model Architecture

### Target Model

- `Qwen3ForCausalLM` loaded from HuggingFace (e.g. `Qwen/Qwen3-8B`), frozen throughout training and inference.
- A subset of intermediate hidden states is extracted at evenly-spaced layer indices (`target_layer_ids`), chosen to span the network depth.
- At inference, `flex_attention` with a custom `score_mod` enforces tree-causal masking: each candidate token attends only to its ancestors in the draft tree (plus the full context prefix).

### Draft Model

`DFlashDraftModel` is a small Qwen3-based transformer (`num_hidden_layers` ≪ target) with two inputs:

1. **Tree embeddings** (`noise_embds`): embeddings of the tree's token sequence. The root node uses the last accepted token's embedding; all other positions use a mask-token embedding.
2. **Context features** (`target_ctx_features`): hidden states from the target model's selected layers.

**Context projection**: hidden states from `N = len(target_layer_ids)` target layers are concatenated along the feature dimension `[B, S, N·D_target]` and projected to the draft hidden size via `fc: Linear(N·D_target, D_draft, bias=False)` followed by `hidden_norm`.

**Cross-attention**: each `Qwen3DFlashDecoderLayer` uses attention where the KV tensors are the concatenation of projected context features and the tree's own hidden states. This lets every draft token attend to the full context while remaining causally ordered within the tree.

**Optional components**:

| Component | Config flag | Description |
|---|---|---|
| Tree position embeddings | `use_tree_pos_emb` | Learned additive embedding indexed by intra-tree position |
| Tree relation bias | `use_additive_tree_pos_bias` | Per-head additive bias based on ancestor/sibling relation between nodes |
| Confidence head | `use_q_head` | `Linear(D_draft, 1)` outputting per-node acceptance logits for tree pruning |

The draft model shares the target's `lm_head` (tied embeddings) for vocabulary projection.

---

## Tree Structure

The default tree processor (`EveryBranchTreeProcessor`) builds a fixed branching tree of depth `d` with 8 branches per level:

- **Primary branch**: a linear AR chain of depth `d` (root → leaf).
- **Secondary branches**: at each of the 8 branching points along the tree, a subtree of depth 1 with a single alternative token prediction, yielding `7 × d` additional leaf nodes.
- **Total nodes**: `T = 8 × d`.

Each node is characterized by:
- `parent_idx`: index of its parent node (−1 for root)
- `depth`: distance from root
- `seq_position`: absolute sequence position (`curr_pos + depth`)
- `is_leaf`: whether the node has no children
- `relation_map`: encodes the ANCESTOR / DESCENDANT / PARENT / CHILD / SIBLING relationship to every other node (used by the attention bias)

`full_tree_mask[i, j] = True` means node j is an ancestor-or-self of node i (j can influence i's attention context).

Positional encoding uses RoPE applied to `sequence_position_ids` (the absolute sequence positions of all tree nodes), ensuring continuity with the context's positional representation.

---

## Training

### Label Generation

For each training example, ground-truth tree labels are constructed from the frozen target model:

1. **Prefill**: run target on the full context to obtain hidden states at `target_layer_ids` and next-token logits at every position.
2. **Tree labeling**: for each anchor position (a position in the training sequence chosen as a tree root), decode the tree by sampling the top-8 tokens from the target's logits at each branching point. Tokens already selected as ancestors are excluded from subsequent children to encourage diversity.
3. **Probabilities**: record `tree_ar_prob[b, n, t]` = the target model's probability for the selected token at each node.
4. **Cumulative probabilities**: `tree_cum_prob[b, n, t]` = product of `tree_ar_prob` along the path from the root to node t. This represents how likely the target model is to follow that path during AR generation.

**Offline labels**: labels can be pre-computed and stored in HDF5 (`labels_h5_path`). Each training sample then loads labels by `(anchor_sequence_idx, anchor_response_idx)` row indices, skipping the online labeling overhead.

### Drafter Forward Pass

Input: `noise_embds [B, N_T·T, D_draft]` (tree token embeddings) and `target_ctx_features [B, S, D_draft]`.

The drafter processes the full tree in one forward pass using tree-causal attention (each node attends to its ancestors via `tree_info.tree_mask`). Output hidden states `tree_hs [B, N_T·T, D_draft]` are projected by the shared `lm_head` to logits `[B, N_T, T−1, V]` (predicting nodes 1..T−1 from their parents).

### Loss

**Primary — cross-entropy**:

```
L_lm = CrossEntropy(tree_logits, tree_labels[:, :, 1:])
```

During training, the `lm_head` projection and CE can optionally be computed in chunks over
flattened prediction positions via `trainer.ce_chunk_size`. This preserves the loss exactly
while avoiding materializing the full `[B, N_T, T-1, V]` logits tensor at once.

When `loss_weighting="target_probs"`, each token's loss is scaled by the target model's cumulative probability of reaching that node:

```
w[b, n, t] = tree_cum_prob[b, n, t] / sum_t(tree_cum_prob[b, n, t]) × (T−1)
L_lm = sum(L_lm × w)
```

This focuses training on tokens that the target model is likely to generate.

**Secondary — sibling overlap** (optional, `sibling_overlap_loss_weight > 0`): penalizes top-k prediction overlap between sibling branches. Sibling nodes share the same parent, so they should predict diverse alternatives. Controlled by `sibling_overlap_temperature` and `sibling_overlap_topk`.

**Tertiary — q-head** (optional, `use_q_head=True`): binary cross-entropy between q-head logits and `is_correct` (whether the drafter's argmax matches the tree label at each node):

```
L_q = BinaryCrossEntropy(q_head(tree_hs), is_correct.float())
```

**Total**:

```
L = L_lm + sibling_overlap_loss_weight × L_sibling + 0.5 × L_q
```

**Optimizer**: AdamW with cosine LR decay and linear warmup over `warmup_steps`.

---

## Speculative Decoding

### Prefill

Run the target model on the input prompt to populate its KV cache and obtain `target_context_features` (hidden states at `target_layer_ids`). Sample the first output token.

### Per-Step Loop

Each iteration advances `curr_pos` by `acceptance_length + 1` tokens.

**1. Build tree inputs**

Construct `inference_extras`: noise embeddings for the tree rooted at the last accepted token (root embedding = that token; all other positions = mask token embedding). Compute `sequence_position_ids` as `curr_pos + intra_tree_depth`.

**2. Drafter forward**

```
drafter_logits = lm_head(DFlashDraftModel(noise_embds, target_ctx_features))  # [1, T, V]
drafter_preds  = argmax(drafter_logits)  # [1, N_T, T], greedy
draft_probs    = softmax(drafter_logits)[drafter_preds]  # [1, N_T, T]
```

The root slot of `drafter_preds` is overwritten with the last accepted token; its draft probability is set to 1.0.

**3. Tree pruning (optional)**

If `n_candidate_tokens` is set, score each node by the cumulative product of q-values along its ancestor path (estimated acceptance probability). Keep the top-k nodes by this score and remap parent indices into the pruned index space.

**4. Verifier forward**

Run the target on `candidate_extras.input_ids [1, T']` using tree-masked `flex_attention`:

```python
score_mod(score, B, H, Q, KV):
    if KV < curr_pos:
        return score  # attend to full context
    elif tree_masks[B, Q, KV - curr_pos]:
        return score  # attend to ancestors in tree
    else:
        return -inf
```

This produces `verifier_out.logits [1, T', V]`, the target's next-token distributions conditioned on each candidate's ancestor path.

**5. Rejection sampling**

For each non-root candidate token j with draft probability `p_draft[j]`:

```
p_target[j] = softmax(verifier_logits[parent[j]] / temperature)[candidate_token[j]]
acceptance_ratio[j] = min(1, p_target[j] / p_draft[j])
token_accepted[j] = Bernoulli(acceptance_ratio[j])
```

**6. Path selection**

A path ending at vertex v is accepted if and only if every ancestor token is individually accepted:

```
path_accepted[v] = all(token_accepted[j] for j in ancestors(v))
best_vertex = argmax_v(path_accepted[v] × depth[v])
acceptance_length = depth[best_vertex]
```

**7. Commit and cache update**

- Write `drafter_preds` along the accepted path to `output_ids[curr_pos : curr_pos + acceptance_length]`.
- Write the verifier's sampled prediction at `best_vertex` as a bonus token at `output_ids[curr_pos + acceptance_length]`.
- Trim the target KV cache: discard all tree positions except those on the accepted path.
- Slice `target_context_features` to the accepted positions for the next iteration's drafter input.
- Advance: `curr_pos += acceptance_length + 1`.

---

## Configuration Reference

| Option | Default | Description |
|---|---|---|
| `depth` | — | Tree depth; total tree size = `8 × depth` |
| `n_candidate_tokens` | `None` | Max candidates sent to verifier; `None` = full tree |
| `loss_weighting` | `None` | `"target_probs"` to weight loss by path probability |
| `ce_chunk_size` | `None` | Number of prediction positions per LM-head / CE chunk during training |
| `sibling_overlap_loss_weight` | `0.0` | Diversity regularization weight |
| `sibling_overlap_temperature` | `0.5` | Softmax temperature for overlap computation |
| `sibling_overlap_topk` | `8` | Top-k predictions considered per node |
| `target_temperature` | `1.0` | Verifier sampling temperature (affects rejection sampling) |
| `use_q_head` | `False` | Enable confidence head for tree pruning |
| `labels_h5_path` | `None` | Path to pre-computed HDF5 label file |
| `lr` | `6e-4` | Peak learning rate |
| `warmup_steps` | `128` | Linear LR warmup steps |
| `grad_accum_steps` | `1` | Gradient accumulation |
| `precision` | `"bf16-mixed"` | Training precision |
