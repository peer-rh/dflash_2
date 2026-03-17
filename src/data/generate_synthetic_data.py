import os
import json
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

from vllm import LLM, SamplingParams

def load_dflash_prompts(tokenizer, num_samples=800000, max_seq_len=3072):
    """Loads the datasets mentioned in the paper: Nemotron V2 & CodeAlpaca."""
    print("Loading Nemotron and CodeAlpaca datasets...")
    prompts = []
    
    # 1. Load CodeAlpaca
    try:
        code_ds = load_dataset("HuggingFaceH4/CodeAlpaca_20k", split="train")
        for row in code_ds:
            content = row.get("prompt", "")
            prompts.append([{"role": "user", "content": content}])
    except Exception as e:
        print(f"Error loading CodeAlpaca: {e}")

    # 2. Load Nemotron V2 (Streaming to avoid downloading the entire 6M dataset)
    try:
        nemo_ds = concatenate_datasets(load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split=["chat", "math", "code", "stem"]))
        nemo_ds = nemo_ds.shuffle(seed=42)  
        for row in nemo_ds:
            if 'messages' in row and len(row['messages']) > 0:
                message = next(m for m in row['messages'] if m['role'] == 'user')
                if len(message['content']) >= 6144:
                    continue
                prompts.append([{"role": "user", "content": message['content']}])
            
            # Stop once we have slightly more than needed (buffer for shuffling)
            if len(prompts) >= num_samples * 2:
                break
    except Exception as e:
        print(f"Error loading Nemotron: {e}")

    # Shuffle and limit to the requested dataset size
    random.seed(42)
    random.shuffle(prompts)
    # prompts = prompts[:num_samples]

    print("Applying target model's chat template...")
    formatted_prompts = []
    for p in prompts:
        formatted_prompts.append(
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        )
    lens = [len(i) for i in tokenizer(formatted_prompts, add_special_tokens=False)['input_ids']]
    print("Total prompts loaded:", len(formatted_prompts))
    formatted_prompts = [p for p, l in zip(formatted_prompts, lens) if l <= max_seq_len * 0.66][:num_samples]
    print(f"Prompts after filtering by length: {len(formatted_prompts)}")
    return formatted_prompts

def generate_texts(args):
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    prompts = load_dflash_prompts(tokenizer, num_samples=args.num_samples, max_seq_len=args.max_seq_len)
    if args.num_shards > 1:
        chunk_size = len(prompts) // args.num_shards
        start_idx = args.shard_idx * chunk_size
        end_idx = start_idx + chunk_size if args.shard_idx < args.num_shards - 1 else len(prompts)
        prompts = prompts[start_idx:end_idx]
        
        # Append shard ID to filename (e.g., text_data_shard0.jsonl)
        # base, ext = os.path.splitext(args.text_output_file)
        # args.text_output_file = f"{base}_shard{args.shard_idx}{ext}"
        print(f"Worker {args.shard_idx}: Processing {len(prompts)} prompts...")
    
    print(f"Initializing vLLM engine for {args.target_model}...")
    llm = LLM(
        model=args.target_model, 
        tensor_parallel_size=args.tp_size, 
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=args.max_seq_len,
    )
    
    # DFlash uses a max of 2048 generated tokens for targets
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_gen_tokens
    )
    
    print("Starting fast batched generation...")
    outputs = llm.generate(prompts, sampling_params)
    
    os.makedirs(os.path.dirname(args.text_output_file) or ".", exist_ok=True)
    with open(args.text_output_file, "w", encoding="utf-8") as f:
        for output in outputs:
            response_text = output.outputs[0].text
            f.write(json.dumps({
                "prompt": output.prompt,
                "response": response_text,
                "full_text": output.prompt + response_text
            }) + "\n")
            
    print(f"✅ Text dataset saved to {args.text_output_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DFlash Training Dataset")
    parser.add_argument("--target-model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Target model ID")
    parser.add_argument("--num-samples", type=int, default=800000, help="Number of samples to process")
    parser.add_argument("--max-seq-len", type=int, default=3072, help="Max sequence length (Appendix A.1 specifies 3072)")
    parser.add_argument("--max-gen-tokens", type=int, default=2048, help="Max generated tokens (Table 1)")
    parser.add_argument("--text-output-file", type=str, default="./dflash_data/text_data.jsonl")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor Parallel size for vLLM")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of GPU workers")
    parser.add_argument("--shard-idx", type=int, default=0, help="Index of this specific worker")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    
    
    args = parser.parse_args()

    generate_texts(args)
