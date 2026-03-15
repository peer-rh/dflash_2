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

uv run -m src.trainer --run_name $EXPERIMENT_NAME --seed 42 --trainer.dev_run true \
    --drafter "z-lab/Qwen3-4B-DFlash-b16" \
    --target $TARGET_MODEL \
    --data.data_path ../dflash/datasets/qwen3-4b/ \
    --data.batch_size 2 --data.seq_len 2048 --data.n_blocks 32 --data.block_size 24 \
    --data.num_workers 4 --trainer.checkpoint_path $checkpoint_path \
    --trainer.grad_accum_steps 4 --trainer.log_every 10 --trainer.num_epochs 32  --trainer.eval_every 1024 \
    --tree_type block --tree_args "{'block_size': 16}" --trainer.ddp false --trainer.precision 'bf16-true'
