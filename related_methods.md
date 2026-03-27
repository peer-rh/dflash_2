# Related Methods

## Speculative Decoding

This note focuses on methods that are close to DFlash in spirit: speculative decoding systems that
either improve the drafter itself or improve how draft candidates are organized and verified,
especially with tree-structured verification.

### EAGLE-3

- **Paper**: [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://openreview.net/forum?id=C9NEblP8vS)
- **Core idea**: EAGLE-3 replaces EAGLE-style next-feature prediction with direct token prediction,
  fuses multiple target layers instead of relying on only the top layer, and uses a
  "training-time test" recipe so the training and test procedures match more closely.
- **Draft form**: autoregressive draft model conditioned on fused target features. In the acceptance
  rate analysis, the paper uses a chain-like draft rather than a tree.
- **Target models reported**:
  - Vicuna-13B
  - Llama-Instruct-3.1-8B
  - Llama-Instruct-3.3-70B
  - DeepSeek-R1-Distill-Llama-8B
- **Benchmarks / datasets reported**:
  - MT-Bench
  - HumanEval
  - GSM8K
  - Alpaca
  - CNN/DailyMail
- **Training data note**:
  - Figure 2 reports scaling experiments on Llama-Instruct-3.1-8B with data scale measured relative
    to ShareGPT.
- **Headline results reported**:
  - Up to **6.47x** speedup on HumanEval with Vicuna-13B at temperature 0.
  - Up to **5.58x** on MT-Bench and **5.32x** on GSM8K with Vicuna-13B at temperature 0.
  - The abstract reports up to **6.5x** speedup overall and a **1.38x** throughput improvement in
    SGLang at batch size 64.
- **Why it matters for DFlash**:
  - EAGLE-3 is a strong feature-conditioned speculative baseline, but it is still fundamentally
    sequence-style drafting. DFlash differs by predicting and verifying a whole tree in one pass.

### DFlash

- **Source in this repo**: [method.md](./method.md)
- **Core idea**: DFlash is a tree-based speculative decoding system for Qwen3 models. A small draft
  model takes target hidden-state features plus tree inputs, proposes a tree of candidate tokens in
  one forward pass, and the frozen target verifies the whole tree with tree-causal masked attention.
- **Draft form**:
  - Small Qwen3-based draft model with cross-attention into selected target hidden states.
  - Shared target `lm_head`.
  - Optional tree position embeddings, relation bias, and q-head.
- **Tree form**:
  - The default `EveryBranchTreeProcessor` builds a fixed `8 x depth` tree with one primary chain
    and multiple side branches.
  - The repo also contains `BlockTree` and `PrunableTreeProcessor` variants.
- **Target / draft model configurations currently present in the repo**:
  - [experiments/qwen_3_4b_every_branch/run.sh](./experiments/qwen_3_4b_every_branch/run.sh):
    `Qwen/Qwen3-4B` target with a 3-layer draft model.
  - [experiments/qwen_3_0_6b_every_branch/run.sh](./experiments/qwen_3_0_6b_every_branch/run.sh):
    `Qwen/Qwen3-0.6B` target with a 3-layer draft model.
- **Training data / datasets currently present in the repo**:
  - [src/data/generate_synthetic_data.py](./src/data/generate_synthetic_data.py) says the prompt
    generator uses:
    - `nvidia/Nemotron-Post-Training-Dataset-v2`
    - `HuggingFaceH4/CodeAlpaca_20k`
  - Training runs in the experiment scripts also reference:
    - `peerrh/q3_4b_100k`
    - precomputed tree labels such as `datasets/q3_4b_100k_stage2.h5`
- **Evaluation / benchmark datasets supported by the repo**:
  - Default quality datasets in [src/data/data_module.py](./src/data/data_module.py):
    - GSM8K
    - Alpaca
    - HumanEval
  - Additional supported eval datasets in [src/data/eval_data.py](./src/data/eval_data.py):
    - MT-Bench
    - MATH-500
    - AIME 2024 / 2025
    - MBPP
    - LBPP
    - SWE-bench Lite
    - LiveCodeBench
- **Benchmark/result note**:
  - This repo currently documents the method and experiment configurations, but it does not yet
    contain a single paper-style summary table with final headline speedups.
- **Why it matters**:
  - Compared with sequence-style speculation, DFlash is much closer to the tree-structured line of
    work because it drafts and verifies many tree nodes in parallel, not just a chain of future
    tokens.

### SpecInfer / Collie

- **Paper / arXiv version**: [Collie: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781)
- **Core idea**: organize candidates from one or more small speculative models into a **token tree**
  and verify the whole tree in parallel with the target model, instead of verifying multiple
  independent sequences one by one.
- **Draft form**:
  - multiple small speculative models in the original system description;
  - tree verification is the key systems contribution.
- **Target / draft models explicitly visible in the paper figures**:
  - LLaMA-7B target with LLaMA-160M small models
  - LLaMA-30B target with LLaMA-160M small models
- **Benchmarks / datasets explicitly visible in the paper figures**:
  - Alpaca prompts
- **Hardware setups highlighted in the paper figures**:
  - single NVIDIA A10 for the LLaMA-7B experiment
  - four NVIDIA A10 GPUs for the LLaMA-30B serving experiment
- **Headline results visible from the figure captions**:
  - The paper reports lower latency and higher throughput than sequence-based speculative decoding
    under its Alpaca serving setup.
  - The figures emphasize that tree verification avoids the memory overhead of verifying many
    independent sequences.
- **Why it matters for DFlash**:
  - SpecInfer is one of the earliest clear demonstrations that tree verification is a systems win,
    even before later work began optimizing the tree shape itself.

### Sequoia

- **Paper**: [Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding](https://arxiv.org/abs/2402.12374)
- **Core idea**:
  - use dynamic programming to construct a better token tree;
  - change the sampling / verification procedure to be more robust across temperatures;
  - add a hardware-aware optimizer that picks tree size and depth for a given platform.
- **Tree contribution**:
  - explicitly positions itself as a scalable tree-construction method;
  - builds on and improves the verification logic used in earlier tree methods such as SpecInfer.
- **Target / draft models reported**:
  - On-chip:
    - Llama2-7B with JackFram/Llama-68M
    - Llama2-13B with JackFram/Llama-68M or JackFram/Llama-160M
    - Vicuna-33B with Sheared-Llama-1.3B
  - Offloading:
    - Llama2-70B target with Llama2-7B draft
- **Benchmarks / datasets reported**:
  - C4(en) validation
  - OpenWebText
  - CNN/DailyMail
- **Hardware reported**:
  - A100 and L40 for on-chip settings
  - L40 for offloading experiments
- **Headline results reported**:
  - Up to **4.04x** speedup for Llama2-7B on A100.
  - Up to **3.73x** speedup for Llama2-13B on A100.
  - Up to **2.27x** speedup for Vicuna-33B on A100.
  - For offloaded Llama2-70B on L40, as low as **0.56 s/token**, about **9.96x** over the paper's
    optimized offloading baseline.
- **Why it matters for DFlash**:
  - Sequoia is one of the cleanest "tree optimization" baselines to compare against when the
    research question is whether better tree construction, better verification, or better drafter
    training gives the largest gain.

### OPT-Tree

- **Paper**: [OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure](https://arxiv.org/abs/2406.17276)
- **Core idea**: choose the draft tree **adaptively at each decoding step** to maximize the expected
  acceptance length, instead of using a fixed heuristic tree.
- **Tree contribution**:
  - the tree changes from step to step;
  - the paper frames tree construction as a direct optimization over expected accepted length.
- **Target models reported**:
  - Llama-2-7B
  - Llama-2-13B
  - Llama-2-70B
  - Vicuna-33B
- **Draft models reported**:
  - one or two smaller same-family models per target
  - corresponding EAGLE draft models
- **Benchmarks / datasets reported**:
  - MT-Bench
  - GSM8K
  - C4 (used to build Sequoia comparison trees in the scaling study)
- **Hardware reported**:
  - RTX 4090 for Llama-2-7B
  - L20 for Llama-2-13B
  - 4x A100-PCIE-40GB for Llama-2-70B and Vicuna-33B
- **Headline results reported**:
  - Up to about **3.2x** speedup over vanilla autoregressive decoding.
  - With Llama-2-70B + Llama-2-7B, the paper reports that sufficiently large OPT-Trees can reach
    accepted lengths around **10 tokens** in a single decoding step.
- **Why it matters for DFlash**:
  - OPT-Tree is a natural comparison point if the research question is whether a **fixed learned
    tree** (like DFlash) or a **step-adaptive tree** gives better speed/acceptance trade-offs.

## Takeaways For DFlash Research

- **EAGLE-3** is the strongest nearby non-tree baseline here: feature-conditioned, high acceptance,
  and very strong speedups on common LLM benchmarks.
- **SpecInfer** is the key early systems paper showing that verifying a token tree in parallel is
  worthwhile.
- **Sequoia** and **OPT-Tree** are the most relevant tree-construction baselines:
  - Sequoia focuses on scalable tree optimization and hardware-aware tuning.
  - OPT-Tree focuses on adaptive per-step tree construction.
- **DFlash** is differentiated less by "tree verification exists" and more by **how the drafter is
  trained and conditioned**:
  - target hidden-state fusion,
  - one-pass tree prediction,
  - optional relation bias / q-head,
  - exact target-distribution verification via rejection sampling.

## Sources

- EAGLE-3 paper: <https://openreview.net/forum?id=C9NEblP8vS>
- EAGLE-3 PDF lines used for benchmarks/results: <https://openreview.net/pdf/28c4c8cf58b0086a2136d73f6059ada87ac33e53.pdf>
- DFlash method note in this repo: [method.md](./method.md)
- DFlash training data generator in this repo: [src/data/generate_synthetic_data.py](./src/data/generate_synthetic_data.py)
- DFlash eval dataset loader in this repo: [src/data/eval_data.py](./src/data/eval_data.py)
- DFlash data module defaults in this repo: [src/data/data_module.py](./src/data/data_module.py)
- DFlash experiment configs in this repo:
  - [experiments/qwen_3_4b_every_branch/run.sh](./experiments/qwen_3_4b_every_branch/run.sh)
  - [experiments/qwen_3_0_6b_every_branch/run.sh](./experiments/qwen_3_0_6b_every_branch/run.sh)
  - [experiments/q3_4b_exp_1/run.sh](./experiments/q3_4b_exp_1/run.sh)
- SpecInfer / Collie paper: <https://arxiv.org/abs/2305.09781>
- Sequoia paper: <https://arxiv.org/abs/2402.12374>
- OPT-Tree paper: <https://arxiv.org/abs/2406.17276>
