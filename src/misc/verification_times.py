import argparse
import time

import pandas as pd
import torch
from transformers.cache_utils import DynamicCache

from ..models.qwen3 import Qwen3ForCausalLM


MODELS = [
    "qwen/qwen3-4b",
    "qwen/qwen3-8b",
    "qwen/qwen3-14b",

]
BSZ = [1, 2, 4, 8, 16]
TREE_SIZE = [16, 24, 32, 64, 128]
KV_LEN = [512, 1024, 2048]

N_WARMUP = 10
N_TEST = 100
DTYPE = torch.bfloat16



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DFlash drafting latencies.")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv-path", type=str, default=None)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def build_random_cache(
    model: Qwen3ForCausalLM,
    bsz:int, 
    kv_len: int,
    device: torch.device,
) -> DynamicCache:
    cache = DynamicCache(config=model.config)
    hidden_states = torch.randn(
        bsz,
        kv_len,
        model.config.hidden_size,
        device=device,
        dtype=DTYPE,
    )
    position_ids = torch.arange(kv_len, device=device).unsqueeze(0)
    model(
        inputs_embeds=hidden_states,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    return cache


@torch.inference_mode()
def benchmark_case(
    model: Qwen3ForCausalLM,
    bsz: int,
    tree_size: int,
    kv_len: int,
    n_warmup: int,
    n_test: int,
    device: torch.device,
) -> dict[str, float | int | str]:

    if model.config._attn_implementation != "flex_attention":
        raise RuntimeError(
            f"Expected flex attention, got {model.config._attn_implementation!r}."
        )

    cache = build_random_cache(model, bsz=bsz, kv_len=kv_len, device=device)
    verifier_hidden_states = torch.randn(
        bsz,
        tree_size,
        model.config.hidden_size,
        device=device,
        dtype=DTYPE,
    )
    position_ids = torch.arange(kv_len, kv_len + tree_size, device=device).unsqueeze(0)

    for _ in range(n_warmup):
        model(
            inputs_embeds=verifier_hidden_states,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        cache.crop(kv_len)
    synchronize(device)

    timings_ms: list[float] = []
    for _ in range(n_test):
        synchronize(device)
        start = time.perf_counter()
        model(
            inputs_embeds=verifier_hidden_states,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
        cache.crop(kv_len)

    result = {
        "tree_size": tree_size,
        "kv_len": kv_len,
        "bsz": bsz,
        "verifiertime_ms": sum(timings_ms) / len(timings_ms),
        "verifiertime_std_ms": pd.Series(timings_ms).std(ddof=0),
        "attn_implementation": model.config._attn_implementation,
    }

    del cache
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main() -> None:
    args = parse_args()

    if args.device != "cuda":
        raise RuntimeError("Use a CUDA device to benchmark with flex attention.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no CUDA device is available.")

    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    results = []
    for model_name in MODELS:
        model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            attn_implementation="flex_attention"
        ).to(device)
        model.eval()
        for bsz in BSZ:
            for tree_size in TREE_SIZE:
                for kv_len in KV_LEN:
                    result = benchmark_case(
                        model=model,
                        bsz=bsz,
                        tree_size=tree_size,
                        kv_len=kv_len,
                        n_warmup=args.n_warmup,
                        n_test=args.n_test,
                        device=device,
                    )
                    result["model_name"] = model_name
                    result["bsz"] = bsz
                    print(result)
                    results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\nResults:")
    print(df.to_string(index=False))

    if args.csv_path is not None:
        df.to_csv(args.csv_path, index=False)
        print(f"\nSaved results to {args.csv_path}")


if __name__ == "__main__":
    main()
