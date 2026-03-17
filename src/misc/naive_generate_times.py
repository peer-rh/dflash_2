import argparse
import json
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DynamicCache

try:
    from ..data.data_module import setup_quality_dataset
    from ..models.qwen3 import Qwen3ForCausalLM
except ImportError:
    from src.data.data_module import setup_quality_dataset
    from src.models.qwen3 import Qwen3ForCausalLM


DEFAULT_QUALITY_DATASETS = ["gsm8k", "alpaca", "humaneval"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure naive generation TPS on the same quality data used in validate_quality."
    )
    parser.add_argument("--target", type=str, required=True, help="Target model path or HF id")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--attn-implementation", type=str, default="flex_attention")
    parser.add_argument(
        "--quality-datasets",
        type=str,
        nargs="+",
        default=DEFAULT_QUALITY_DATASETS,
    )
    parser.add_argument("--n-samples-per-quality-dataset", type=int, default=16)
    return parser.parse_args()


def sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def wall_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def naive_generate(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: int | None,
) -> SimpleNamespace:
    output_ids = input_ids
    past_key_values = DynamicCache(config=model.config)

    start_time = wall_time(input_ids.device)
    post_prefill_time = None

    for step in range(max_new_tokens):
        new_input_ids = output_ids[:, past_key_values.get_seq_length() :]
        out = model(
            input_ids=new_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = sample(out.logits[:, -1, :], temperature=temperature)
        output_ids = torch.cat((output_ids, next_token), dim=1)
        past_key_values = out.past_key_values

        if step == 0:
            post_prefill_time = wall_time(input_ids.device)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    done_time = wall_time(input_ids.device)
    generated_ids = output_ids[:, input_ids.shape[1] :]
    generated_tokens = generated_ids.shape[1]

    total_time = done_time - start_time
    decode_only_time = (
        max(done_time - post_prefill_time, 1e-12)
        if post_prefill_time is not None
        else max(total_time, 1e-12)
    )

    return SimpleNamespace(
        output_ids=generated_ids,
        generated_tokens=generated_tokens,
        total_time=total_time,
        decode_only_time=decode_only_time,
        tps=generated_tokens / decode_only_time if generated_tokens > 0 else 0.0,
        tps_with_prefill=generated_tokens / total_time if generated_tokens > 0 else 0.0,
    )


def build_quality_loaders(
    quality_datasets: list[str],
    seed: int,
    n_samples_per_quality_dataset: int,
) -> dict[str, DataLoader]:
    quality_ds = setup_quality_dataset(
        quality_datasets=quality_datasets,
        seed=seed,
        n_samples_per_quality_dataset=n_samples_per_quality_dataset,
    )
    return {
        name: DataLoader(ds, batch_size=1, shuffle=False)
        for name, ds in quality_ds.items()
    }


@torch.inference_mode()
def benchmark_split(
    split_name: str,
    loader: DataLoader,
    model: Qwen3ForCausalLM,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, float | int | str]:
    info = {
        "tps": [],
        "tps_with_prefill": [],
        "generated_tokens": [],
        "decode_only_times": [],
        "total_times": [],
    }

    for sample in loader:
        messages: list[dict[str, str]] = []

        for user_content in sample["turns"]:
            user_text = user_content[0]
            print(f"User: {user_text}")

            messages.append({"role": "user", "content": user_text})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

            outputs = naive_generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )

            info["tps"].append(outputs.tps)
            info["tps_with_prefill"].append(outputs.tps_with_prefill)
            info["generated_tokens"].append(outputs.generated_tokens)
            info["decode_only_times"].append(outputs.decode_only_time)
            info["total_times"].append(outputs.total_time)

            output_text = tokenizer.decode(outputs.output_ids[0], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})

    total_generated_tokens = sum(info["generated_tokens"])
    total_decode_only_time = sum(info["decode_only_times"])
    total_time = sum(info["total_times"])

    return {
        "split": split_name,
        "num_generations": len(info["tps"]),
        "generated_tokens_total": total_generated_tokens,
        "tps_mean": sum(info["tps"]) / len(info["tps"]) if info["tps"] else 0.0,
        "tps_with_prefill_mean": (
            sum(info["tps_with_prefill"]) / len(info["tps_with_prefill"])
            if info["tps_with_prefill"]
            else 0.0
        ),
        "aggregate_tps": (
            total_generated_tokens / total_decode_only_time
            if total_decode_only_time > 0
            else 0.0
        ),
        "aggregate_tps_with_prefill": (
            total_generated_tokens / total_time
            if total_time > 0
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("medium")

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    model = Qwen3ForCausalLM.from_pretrained(
        args.target,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model = model.to(device)
    model.eval()

    quality_loaders = build_quality_loaders(
        quality_datasets=args.quality_datasets,
        seed=args.seed,
        n_samples_per_quality_dataset=args.n_samples_per_quality_dataset,
    )

    results = []
    for split_name, loader in quality_loaders.items():
        result = benchmark_split(
            split_name=split_name,
            loader=loader,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    overall_generations = sum(x["num_generations"] for x in results)
    overall_tokens = sum(x["generated_tokens_total"] for x in results)

    print(json.dumps(
        {
            "num_splits": len(results),
            "num_generations": overall_generations,
            "generated_tokens_total": overall_tokens,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
