#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=outputs/logs/%j.out # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=outputs/logs/%j.err # where to store error messages
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=tikgpu10 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='geforce_rtx_3090'
#CommentSBATCH --constraint='rtx_a6000'
#CommentSBATCH --constraint='a100'

# Exit on errors
set -o errexit
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

ETH_USERNAME=prheinboldt
PROJECT_NAME=dflash_2
DIRECTORY="/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}"

cd $DIRECTORY

checkpoint_path="$OUTPUT_DIR/checkpoints"
mkdir -p $checkpoint_path

TARGET_MODEL="Qwen/Qwen3-4B"
MODEL_JSON=$(cat <<EOF
{
  "architectures": [
    "DFlashDraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoModel": "dflash.DFlashDraftModel"
  },
  "block_size": 16,
  "bos_token_id": 151643,
  "dflash_config": {
    "mask_token_id": 151669,
    "target_layer_ids": [
      1,
      9,
      17,
      25,
      33
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 9728,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 3,
  "num_key_value_heads": 8,
  "num_target_layers": 36,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936,
  "use_tree_pos_emb": false,
  "max_tree_size": 32
}
EOF
)
TREE_JSON=$(cat <<EOF
{
    "random_embds": false,
    "block_size": 16,
}
EOF
)

uv run -m src.trainer --run_name $EXPERIMENT_NAME --seed 42 --trainer.compile true --trainer.verbose false \
    --drafter "$MODEL_JSON" \
    --target $TARGET_MODEL \
    --data.data_path ../dflash/datasets/qwen3-4b/ \
    --data.batch_size 24 --data.seq_len 3072 --data.n_blocks 64 --data.block_size 16 \
    --data.num_workers 4 --trainer.checkpoint_path $checkpoint_path \
    --trainer.grad_accum_steps 3 --trainer.log_every 10 --trainer.num_epochs 8  --trainer.eval_every 2048 --trainer.save_every 2048 \
    --tree_type block --tree_args "$TREE_JSON" --trainer.ddp false --trainer.precision 'bf16-true'
