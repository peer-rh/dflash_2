import argparse
import time
import os
import json
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence, get_args, get_origin
import time

from jsonargparse import ActionConfigFile, ArgumentParser
import jsonargparse
import torch
import torch.nn.functional as F
from lightning import Fabric, seed_everything
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import wandb
from wandb.integration.lightning.fabric import WandbLogger
from lightning.fabric.utilities import AttributeDict

from .trees import TreeProcessor
from .trees.fixed_tree import FixedTreeProcessor
from .trees.block_tree import BlockTree
from .util import SpecializedStaticCache, merge_metrics, sample, wall_time
from .data.data_module import DataModule, DataModuleConfig
from .models.dflash import DFlashDraftModel


@dataclass
class TrainerConfig:
    num_epochs: int = 10
    eval_every: int = 1024
    log_every: int = 10
    save_every: int = 1024
    target_temperature: float = 1.0
    precision: str = "bf16-mixed"
    ddp: bool = False
    lr: float = 6e-4
    warmup_steps: int = 128
    grad_accum_steps: int = 1
    dev_run: bool = False
    checkpoint_path: str = "checkpoints"
    compile: bool = False


class Trainer:
    tree_processor: TreeProcessor
    def __init__(
        self,
        config: TrainerConfig,
        target: str,
        logger: WandbLogger,
        data: DataModuleConfig,
        drafter: dict[str, Any] | str,
        tree_type: Literal["fixed"] = "fixed",
        tree_args: dict[str, Any] | None = None,
    ):
        self.config = config
        self.fabric = Fabric(
            precision=config.precision, # type: ignore
            strategy="ddp" if config.ddp else 'auto',
            loggers=logger,
        )
        self.fabric.launch()
        self.wandb = logger
        tree_args = tree_args or {}

        self.data_module = DataModule(data, target=target)
        if self.fabric.local_rank == 0:
            self.data_module.preprocess()
        self.fabric.barrier()
        if self.fabric.local_rank != 0:
            self.data_module.preprocess() # This can load the cached preprocessed data
        self.fabric.barrier()

        self.trainloader, self.valloader = self.fabric.setup_dataloaders(
            self.data_module.get_train_dataloader(),
            self.data_module.get_val_dataloader(),
        )
        quality_loaders = self.data_module.get_quality_dataloaders()
        quality_loaders_fabric = self.fabric.setup_dataloaders(
            *quality_loaders.values()
        )
        self.quality_loaders = dict(zip(quality_loaders.keys(), quality_loaders_fabric))

        self.steps_per_epoch = (
            len(self.trainloader) // self.fabric.world_size // config.grad_accum_steps
        )
        self.total_steps = self.steps_per_epoch * config.num_epochs
        self.tokenizer = AutoTokenizer.from_pretrained(target)
        self.lm_head = lambda x: self.target.lm_head(x.to(self.target.dtype)) # type: ignore

        if isinstance(drafter, str):
            self.drafter = DFlashDraftModel.from_pretrained(drafter, attn_implementation="flex_attention")
        else:
            drafter['attn_implementation'] = "flex_attention"
            self.drafter: DFlashDraftModel = DFlashDraftModel(drafter)
        self.mask_token_id = getattr(
            self.drafter, "mask_token_id", self.tokenizer.pad_token_id # type: ignore
        )

        self.optim = torch.optim.AdamW(self.drafter.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optim,
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optim, 4e-2, 1, total_iters=config.warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optim, max(1, self.total_steps - config.warmup_steps)
                ),
            ],
            milestones=[config.warmup_steps],
        )
        self.drafter, self.optim = self.fabric.setup(self.drafter, self.optim)

        target_dtype = (
            torch.bfloat16 if "bf16" in self.config.precision else torch.bfloat16
        )
        self.target = AutoModelForCausalLM.from_pretrained(
            target, dtype=target_dtype, attn_implementation="flex_attention"
        )
        self.target = self.fabric.to_device(self.target)

        if tree_type == "fixed":
            self.tree_processor = FixedTreeProcessor(**tree_args, mask_token_id=self.mask_token_id, device=self.fabric.device)
        elif tree_type == "block":
            self.tree_processor = BlockTree(**tree_args, mask_token_id=self.mask_token_id, device=self.fabric.device)
        else:
            raise ValueError(f"Unsupported tree type: {tree_type}")

        self.global_step = 0
        if self.config.dev_run:
            self.config.num_epochs = 1

    def fit(self):
        for epoch in range(self.config.num_epochs):
            metrics = None
            total_batches = len(self.trainloader)
            for batch_idx, batch in enumerate(self.trainloader, start=1):
                should_step = (
                    batch_idx % self.config.grad_accum_steps == 0
                    or batch_idx == total_batches
                )
                _, this_metrics = self.train_step(
                    batch, is_accumulating=not should_step
                )
                metrics = merge_metrics(metrics, this_metrics)
                if not should_step:
                    continue

                if self.config.dev_run and batch_idx > 20:
                    break

                if self.global_step % self.config.eval_every == 0:
                    self.validate()
                    self.validate_quality()
                if self.global_step % self.config.log_every == 0:
                    print(
                        f"Epoch {epoch} / {self.config.num_epochs} - Step {self.global_step} / {self.steps_per_epoch}"
                    )
                    self.log_metrics(metrics, prefix="train")
                    metrics = None
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()
        print("Fit done; Running final validation...")
        self.validate()
        self.validate_quality()
        self.log_metrics(metrics, prefix="train")
        self.save_checkpoint()

    def save_checkpoint(self):
        state = AttributeDict(
            drafter=self.drafter,
            optimizer=self.optim,
            scheduler=self.scheduler,
            global_step=self.global_step,
            config=self.config,
        )
        self.fabric.save(
            os.path.join(
                self.config.checkpoint_path, f"checkpoint_{self.global_step}.ckpt"
            ),
            state,
        )

    def load_checkpoint(self, checkpoint_path):
        state = AttributeDict(
            drafter=self.drafter,
            optimizer=self.optim,
            scheduler=self.scheduler,
            global_step=self.global_step,
            config=self.config,
        )
        self.fabric.load(checkpoint_path, state)

    @torch.inference_mode()
    def validate(self):
        self.target.eval()
        self.drafter.eval()
        metrics = None
        for batch in self.valloader:
            _, this_metrics = self.process_batch(batch)
            metrics = merge_metrics(metrics, this_metrics)
        self.log_metrics(metrics, prefix="val")
        return metrics

    @torch.inference_mode()
    def validate_quality(self):
        self.target.eval()
        self.drafter.eval()
        for split_name, loader in self.quality_loaders.items():
            info = {
                "acceptance_lengths": [],
                "tps": [],
                "tps_with_prefill": [],
            }
            text = ""
            tps_count = 0
            for s in loader:
                messages: list[dict[str, str]] = []
                for user_content in s["turns"]:
                    user_text = user_content[0]

                    messages.append({"role": "user", "content": user_text})
                    input_text = self.tokenizer.apply_chat_template( 
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    input_ids = self.tokenizer.encode( 
                        input_text, return_tensors="pt"
                    ).to(self.target.device)

                    outputs = self.speculative_generate(
                        input_ids=input_ids, max_length=2048
                    )

                    info["acceptance_lengths"].extend([len(x) for x in outputs.accepted_ids])
                    info["tps"].append(outputs.tps)
                    info["tps_with_prefill"].append(outputs.tps_with_prefill)
                    tps_count += 1

                    output_text = self.tokenizer.decode(
                        outputs.output_ids[0], skip_special_tokens=True
                    )
                    messages.append({"role": "assistant", "content": output_text})

                    text += input_text
                    accepted_texts = self.tokenizer.batch_decode(
                        outputs.accepted_ids, skip_special_tokens=True
                    )
                    extra_texts = self.tokenizer.batch_decode(
                        [[x] for x in outputs.extra_ids], skip_special_tokens=True
                    )
                    for accepted_text, extra_text in zip(
                        accepted_texts,
                        extra_texts,
                    ):
                        text += f"<span style='color: green'>{accepted_text}</span>"
                        text += f"<span style='color: blue'>{extra_text}</span><br>"
            metrics = {
                'acceptance_length_mean': sum(info["acceptance_lengths"]) / len(info["acceptance_lengths"]),
                'tps_mean': sum(info["tps"]) / len(info["tps"]),
                'tps_with_prefill_mean': sum(info["tps_with_prefill"]) / len(info["tps_with_prefill"]),
            }
            metrics_reduced = self.fabric.all_reduce(metrics, reduce_op="mean")
            if self.fabric.is_global_zero:
                # Histogram + examples only for rank 0; else we aggregate over all gpus
                self.wandb.experiment.log(
                {
                    f"{split_name}/acceptance_length_histogram": wandb.Histogram(
                        info["acceptance_lengths"]
                    ),
                    f"{split_name}/examples": wandb.Html(text),
                    f"{split_name}/throughput": metrics_reduced["tps_mean"], # type: ignore
                    f"{split_name}/throughput_with_prefill": metrics_reduced["tps_with_prefill_mean"], # type: ignore
                    f"{split_name}/acceptance_length": metrics_reduced["acceptance_length_mean"], # type: ignore
                },
                step=self.global_step,
            )

    def process_batch(self, batch):
        input_ids = batch["input_ids"]  # [B, S]
        anchors = batch["anchors"]  # [B, N_T]
        document_mask = batch["document_mask"]  # [B, S]
        position_ids = batch["position_ids"]  # [B, S]
        B, S = input_ids.shape

        with torch.no_grad():
            tree_extras = self.tree_processor.construct_training_extras(
                input_ids, anchors, document_mask, position_ids, self.target
            )

            tree_labels = tree_extras.tree_labels  # [B, N_T, T]
            B, N_T, T = tree_labels.shape

            # Run Drafter
            def mask_mod(B, _H, Q, KV):
                Q_TREE = Q // T
                KV_TREE = ((KV - S) // T)
                is_context = KV < S
                is_causal = KV < anchors[B, Q_TREE]
                is_same_doc = (
                    document_mask[B, KV % S] == document_mask[B, anchors[B, Q_TREE]]
                )

                is_same_tree = Q_TREE == KV_TREE
                return (is_context & is_causal & is_same_doc) | (~is_context & is_same_tree)

            drafter_attention_mask = create_block_mask(
                mask_mod,
                B,
                None,
                N_T * T,
                N_T * T + S,
                device=input_ids.device,
                BLOCK_SIZE=128,
            )
            # tree_pred :: [B, N_T, T, N_VOCAB]
            target_ctx_features = self.drafter.extract_ctx_features(tree_extras.target_hidden_states)
            drafter_position_ids = torch.cat(
                (
                    position_ids,
                    tree_extras.sequence_position_ids.view(B, N_T * T),
                ), dim=1
            )
        if self.config.dev_run:
            print("--")
            print("Drafter Inputs:")
            print("Noise Embeddings:", tree_extras.noise_embds.shape)
            print("Target Context Features:", target_ctx_features.shape)
            print("Attention Mask:", drafter_attention_mask[0])

        tree_hs = self.drafter(
            hidden_states=tree_extras.noise_embds.view(B, N_T * T, -1),
            target_ctx_features=target_ctx_features,
            attention_mask=drafter_attention_mask,
            position_ids=drafter_position_ids,
            tree_position_ids=tree_extras.tree_position_ids.reshape(B, N_T * T) if tree_extras.tree_position_ids is not None else None,
        )
        tree_logits = self.lm_head(tree_hs).view(B, N_T, T, -1)

        loss = F.cross_entropy(
            tree_logits.view(-1, tree_logits.size(-1)), tree_labels.view(-1), reduction="sum"
        )
        if self.config.dev_run:
            print('--')
            print("Process_batch")
            # print("Tree Labels:",)
            # for i in range(tree_labels.shape[2]):
                # print(self.tokenizer.decode(tree_labels[0, 0, tree_extras.tree_masks[0, 0, i].bool()]))

            print("Tree Preds:", self.tokenizer.decode(tree_logits.argmax(dim=-1)[0, 0]))
            print("Loss:", loss.item())

        pred_ids = tree_logits.argmax(dim=-1)
        is_correct = pred_ids[:, :, 1:] == tree_labels[:, :, 1:]

        target_labels_aligned = input_ids.gather(1, 
            tree_extras.sequence_position_ids[:, :, 1:].reshape(B, N_T * (T - 1))
        ).view(B, N_T, T - 1)
        depth = tree_extras.sequence_position_ids[:, :, 1:] - anchors[:, :, None] # [B, N_T, T-1]
        is_accepted = target_labels_aligned == pred_ids[:, :, 1:] # [B, N_T, T-1]
        is_accepted = (
            is_accepted[:, :, None, :] & tree_extras.tree_masks[:, :, 1:, 1:]
        ).sum(dim=-1) == depth # [B, N_T, T-1]
        best = (is_accepted * depth).max(dim=-1)
        acceptance_length = best.values + 1
        if self.config.dev_run:
            print('--')
            print("Acceptance Info:")
            print("Target Labels Aligned:", self.tokenizer.decode(target_labels_aligned[0, 0]))
            print("Is Correct:", is_correct[0, 0])
            print("Is Accepted:", is_accepted[0, 0])
            print("Best:", best.values[0, 0])
            print("Acceptance Length:", acceptance_length[0,0])

        
        metrics = {
            "lm_loss_sum": loss.detach(),
            "batch_count": B,
            "block_count": B * N_T,
            "token_count": B * N_T * T,
            "token_correct_count": is_correct.sum().detach(),
            "accepted_length_sum": acceptance_length.sum().detach(),
        }
        return loss / (B * N_T * T), metrics

    def train_step(self, batch, is_accumulating: bool = True):
        self.drafter.train()
        self.target.eval()
        with self.fabric.no_backward_sync(self.drafter, enabled=is_accumulating):
            loss, metrics = self.process_batch(batch)
            self.fabric.backward(loss)
        if not is_accumulating:
            self.global_step += 1
            self.fabric.clip_gradients(
                self.drafter,
                self.optim,
                max_norm=1.0,
                error_if_nonfinite=False,
            )
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad(set_to_none=True)
        return loss, metrics

    @torch.inference_mode()
    def speculative_generate(self, input_ids: torch.Tensor, max_length: int):
        """
        Args:
            input_ids: [1, S]
            max_length: int
        """
        num_input_tokens = input_ids.shape[1]
        output_ids = torch.zeros(
            (1, num_input_tokens + max_length + 128),
            dtype=torch.long,
            device=input_ids.device,
        )
        accepted_ids = []
        extra_ids = []

        # Prefill the Ta
        past_key_values_drafter = DynamicCache(config=self.drafter.config)
        past_key_values_target = SpecializedStaticCache(self.target.config)

        start_time = wall_time()
        verifier_out = self.target(
            input_ids=input_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens] = sample(
            verifier_out.logits[:, 0, :], temperature=self.config.target_temperature
        )[0]
        target_context_features = self.drafter.extract_ctx_features(
            verifier_out.hidden_states
        )
        post_prefill_time = wall_time() 
        curr_pos = num_input_tokens
        eos_token = self.tokenizer.eos_token_id
        while curr_pos < max_length + num_input_tokens:
            inference_extras = self.tree_processor.construct_inference_extras(
                output_ids[:, :curr_pos+1], self.target
            )
            noise_embds = inference_extras.noise_embds
            B, N_T, T, D = noise_embds.shape
            position_ids = inference_extras.sequence_position_ids
            assert position_ids.shape[1] == 1, (
                "Currently inference only supports a single block"
            )
            position_ids = torch.cat(
                (
                    torch.arange(
                        curr_pos - target_context_features.shape[1],
                        curr_pos,
                        device=position_ids.device,
                    ).unsqueeze(0),
                    position_ids.view(1, -1),
                ),
                dim=1,
            )  # [1, T_prev + N_T * T]
            if self.config.dev_run:
                print("--")
                print("Drafter Inputs:")
                print("Position IDs:", position_ids)
                print("Target Context Features:", target_context_features.shape)
                print("Noise Embeddings:", noise_embds.shape)
                print("Drafter KV Cache Len: ", past_key_values_drafter.get_seq_length())

            drafter_out = self.drafter(
                hidden_states=inference_extras.noise_embds.view(1, N_T * T, D),
                target_ctx_features=target_context_features,
                position_ids=position_ids,
                tree_position_ids=(inference_extras.tree_position_ids.view(1, N_T * T) if inference_extras.tree_position_ids is not None else None),
                past_key_values=past_key_values_drafter,
                use_cache=True,
            )
            drafter_logits = self.lm_head(drafter_out)  # [1, N_T * T, V]
            drafter_preds = sample(drafter_logits, 0.0).view(1, N_T, T)  # [1, N_T, T]
            drafter_preds[:, :, 0] = output_ids[0, curr_pos]
            past_key_values_drafter.crop(curr_pos)  # Discard drafted tokens from cache
            if self.config.dev_run:
                print("--")
                print("Drafter Outputs:")
                print("Preds:", self.tokenizer.decode(drafter_preds[0].flatten()).replace("\n", "\\n"))
                print("Drafter KV Cache Len: ", past_key_values_drafter.get_seq_length())

            candidate_extras = self.tree_processor.construct_candidate_extras(
                drafter_preds,
                inference_extras.sequence_position_ids,
            )
            if self.config.dev_run:
                print("--")
                print("Verifier Inputs:")
                print("Input_ids:", self.tokenizer.decode(candidate_extras.input_ids[0]).replace("\n", "\\n"))
                # print("Tree Map:", candidate_extras.tree_masks[0].to(torch.float))
                print("Position Ids:", candidate_extras.sequence_position_ids[0])
                print("Verifier Cache Len: ", past_key_values_target.get_seq_end())
                print("Verifier Cache To Keep: ", past_key_values_target.layers[0].idx_to_keep)

            def score_mod(score, B, _H, Q, KV):
                is_pred = KV >= curr_pos
                return torch.where(
                    ~is_pred | candidate_extras.tree_masks[B, Q, (KV - curr_pos) % T], score, -torch.inf
                )

            verifier_out = self.target(
                input_ids=candidate_extras.input_ids,
                position_ids=candidate_extras.sequence_position_ids,
                score_mod=score_mod,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            verifier_preds = sample(
                verifier_out.logits[0], temperature=self.config.target_temperature
            )[None, :, 0]  # [1, T']
            verifier_preds_aligned = verifier_preds[
                :, candidate_extras.parents_idx[0, 1:]
            ]  # [1, T' - 1]
            if self.config.dev_run:
                print("--")
                print("Verifier Outputs:")
                print("Verifier Preds:", self.tokenizer.decode(verifier_preds[0]).replace("\n", "\\n"))
                print("Verifier Aligned:", self.tokenizer.decode(verifier_preds_aligned[0]).replace("\n", "\\n"))
                print("candidates:", candidate_extras.input_ids[0, 1:])
                print("verifier:", verifier_preds_aligned[0])

            depth = candidate_extras.sequence_position_ids - curr_pos
            is_equal = candidate_extras.input_ids[:, 1:] == verifier_preds_aligned
            is_equal = (
                is_equal[:, None, :] & candidate_extras.tree_masks[:, :, 1:]
            ).sum(dim=-1) == depth
            best = (is_equal * depth).max(dim=-1)
            best_vertex = best.indices[0]
            acceptance_length = best.values[0].item() + 1
            acceptance_mask = candidate_extras.tree_masks[0, best_vertex]

            positions_to_keep = torch.arange(
                curr_pos,
                curr_pos + candidate_extras.input_ids.shape[1],
                device=input_ids.device,
            )[acceptance_mask]
            past_key_values_target.mark_tree_update(curr_pos, positions_to_keep)
            target_context_features = self.drafter.extract_ctx_features(
                verifier_out.hidden_states
            )[:, acceptance_mask]

            output_ids[:, curr_pos : curr_pos + acceptance_length] = (
                candidate_extras.input_ids[:, acceptance_mask]
            )
            output_ids[:, curr_pos + acceptance_length] = verifier_preds[:, best_vertex]
            accepted_ids.append(
                output_ids[0, curr_pos : curr_pos + acceptance_length].tolist()
            )
            extra_ids.append(output_ids[0, curr_pos + acceptance_length].item())
            curr_pos += acceptance_length
            if self.config.dev_run:
                print("--")
                print("Acceptance Info:")
                print("Best Vertex:", best_vertex.item())
                print("Acceptance Length:", acceptance_length)
                print("Positions to Keep:", positions_to_keep)
                print("Curr Pos:", curr_pos)
                print("Accepted Tokens:", self.tokenizer.decode(candidate_extras.input_ids[0, acceptance_mask], skip_special_tokens=False).replace("\n", "\\n"))
                print("Next Token:", self.tokenizer.decode(verifier_preds[:, best_vertex][0], skip_special_tokens=False).replace("\n", "\\n"))
                print("Output IDs:", self.tokenizer.decode(output_ids[0, num_input_tokens:curr_pos+1], skip_special_tokens=False).replace("\n", "\\n"))

            if (output_ids[0, num_input_tokens:curr_pos+1] == eos_token).any():
                # print("DONE", num_input_tokens, curr_pos)
                break


        done_time = wall_time()
        time_with_prefill = done_time - start_time
        time_without_prefill = done_time - post_prefill_time
        tokens_generated = curr_pos - num_input_tokens
        return SimpleNamespace(
            output_ids=output_ids[:, num_input_tokens: curr_pos],
            accepted_ids=accepted_ids,
            extra_ids=extra_ids,
            tps=tokens_generated / time_without_prefill,
            tps_with_prefill=tokens_generated / time_with_prefill,
        )

    def log_metrics(self, metrics, prefix=None):
        reduced_metrics: dict[str, float] = self.fabric.all_reduce(metrics, reduce_op="sum") # type: ignore
        if not self.fabric.is_global_zero:
            return

        metrics = {
            "lm_loss" : reduced_metrics['lm_loss_sum'] / reduced_metrics['token_count'], 
            "total_accuracy": reduced_metrics['token_correct_count'] / reduced_metrics['token_count'], 
            "accepted_length": reduced_metrics['accepted_length_sum'] / reduced_metrics['block_count'],
        }
        # TODO: Support this
        # for k, v in reduced_metrics.items():
        #     if k.startswith("pos_wise_correct_count"):
        #         identifier = k.split("/")[1]
        #         metrics[f"accuracy/{identifier}"] = v / reduced_metrics[f'pos_wise_token_count/{identifier}']
        
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.fabric.log_dict(metrics, step=self.global_step)

    def naive_generate(self, input_ids: torch.Tensor, max_length: int):
        output_ids = input_ids
        past_key_values = DynamicCache()
        start = wall_time()
        for _ in range(max_length):
            out = self.target(input_ids=output_ids[:, past_key_values.get_seq_length():], past_key_values=past_key_values, use_cache=True)
            next_token = sample(out.logits[:, -1], temperature=self.config.target_temperature).unsqueeze(0)
            output_ids = torch.cat((output_ids, next_token), dim=1)
            past_key_values = out.past_key_values
        
        return SimpleNamespace(
            output_ids=output_ids[:, input_ids.shape[1]:],
            tps_with_prefill=(output_ids.shape[1] - input_ids.shape[1]) / (wall_time() - start),
        )

def build_parser() -> ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--target", type=str, required=True, help="Frozen verifier model")
    parser.add_argument("--drafter", type=str, default={})
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only run validation without training",
    )
    parser.add_argument( "--run_name", type=str, required=True, help="Run name for logging")
    parser.add_class_arguments(TrainerConfig, "trainer")
    parser.add_class_arguments(DataModuleConfig, "data")
    parser.add_argument('--tree_type', type=str, default="fixed", help="Type of tree structure to use")
    parser.add_argument('--tree_args', type=dict[str, Any], default={}, help="Arguments for tree processor")
    parser.add_argument("--only-spec-dec", action="store_true", help="Only run speculative decoding without training or validation")
    return parser


def main() -> Trainer:
    parser = build_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    instantiated = parser.instantiate_classes(args)

    try:
        drafter = json.loads(args.drafter)
    except Exception as e:
        print(e)
        drafter = args.drafter

    logger = WandbLogger(
        name=args.run_name,
        project="dflash",
        config=args.as_dict(),
    )
    trainer = Trainer(
        config=instantiated.trainer,
        drafter=drafter,
        target=args.target,
        logger=logger,
        data=instantiated.data,
        tree_type=args.tree_type,
        tree_args=args.tree_args,
    )
    if args.only_spec_dec:
        trainer.validate_quality()
    else:
        trainer.fit()
    logger.experiment.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
