#!/bin/bash

#SBATCH -J finetune-dynaview-lego
#SBATCH -t 60:00:00
#SBATCH --output=finetune-dynaview-lego.out
#SBATCH --error=finetune-dynaview-lego.err
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 4

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_IB_DISABLE=1

NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
MASTER_NODE=$(head -n 1 <<< "$NODELIST")
MASTER_ADDR=$(dig +short $MASTER_NODE)
NODE_COUNT=0
NODE_NUM=($(echo $NODELIST | tr " " "\n" | wc -l))

export PROJECT_ROOT_AT="<your_project_root>"
export INIT_MODEL_PATH="<your_path_to_the_initial_pretrained_model_checkpoint>"
export OUT_DIR="<your_path_to_save_the_finetuned_model_checkpoints_and_resume_from>"
export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
export OMP_NUM_THREADS=1
export WANDB_API_KEY="<your_wandb_api_key>"

for NODE in $NODELIST; do
    if [ "$NODE" == "$MASTER_NODE" ]; then
        srun \
          --nodes=1 --ntasks=1 -w $NODE \
          torchrun --nnodes=$NODE_NUM --node_rank=0 --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=29500 \
            run_bagel_ft.py \
            --dataset_config_file $PROJECT_ROOT_AT/modeling/data/configs/lego.yaml \
            --model_path $INIT_MODEL_PATH \
            --layer_module Qwen2MoTDecoderLayer \
            --max_latent_size 64 \
            --finetune_from_hf True \
            --auto_resume True \
            --resume-model-only True \
            --resume_from $INIT_MODEL_PATH \
            --finetune-from-ema False \
            --log_every 1 \
            --save_every 500 \
            --total_steps 6000 \
            --warmup_steps 300 \
            --lr_scheduler cosine \
            --lr 5e-6 \
            --num_worker 1 \
            --num_shard 32 \
            --expected_num_tokens 16384 \
            --max_num_tokens 36864 \
            --max_num_tokens_per_sample 36864 \
            --prefer_buffer_before 16384 \
            --text_cond_dropout_prob 0 \
            --vae_cond_dropout_prob 0 \
            --vit_cond_dropout_prob 0 \
            --ce_weight 0.0 \
            --img_hist_dropout 0.0 \
            --vwm_attn True \
            --vit_attn_trans True \
            --results_dir finetune-dynaview-lego \
            --checkpoint_dir $OUT_DIR \
            --wandb_api_key $WANDB_API_KEY \
            --wandb_project state-transition-vwm \
            --wandb_name finetune-dynaview-lego &
    else
        ((NODE_COUNT++))
        srun \
          --nodes=1 --ntasks=1 -w $NODE \
          torchrun --nnodes=$NODE_NUM --node_rank=$NODE_COUNT --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=29500 \
            run_bagel_ft.py \
            --dataset_config_file $PROJECT_ROOT_AT/modeling/data/configs/lego.yaml \
            --model_path $INIT_MODEL_PATH \
            --layer_module Qwen2MoTDecoderLayer \
            --max_latent_size 64 \
            --finetune_from_hf True \
            --auto_resume True \
            --resume-model-only True \
            --resume_from $INIT_MODEL_PATH \
            --finetune-from-ema False \
            --log_every 1 \
            --save_every 500 \
            --total_steps 6000 \
            --warmup_steps 300 \
            --lr_scheduler cosine \
            --lr 5e-6 \
            --num_worker 1 \
            --num_shard 32 \
            --expected_num_tokens 16384 \
            --max_num_tokens 36864 \
            --max_num_tokens_per_sample 36864 \
            --prefer_buffer_before 16384 \
            --text_cond_dropout_prob 0 \
            --vae_cond_dropout_prob 0 \
            --vit_cond_dropout_prob 0 \
            --ce_weight 0.0 \
            --img_hist_dropout 0.0 \
            --vwm_attn True \
            --vit_attn_trans True \
            --results_dir finetune-dynaview-lego \
            --checkpoint_dir $OUT_DIR \
            --wandb_api_key $WANDB_API_KEY \
            --wandb_project state-transition-vwm \
            --wandb_name finetune-dynaview-lego &
    fi
done
wait

