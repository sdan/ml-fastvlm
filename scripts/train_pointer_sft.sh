#!/usr/bin/env bash
set -euo pipefail

# Full SFT (LM loss on, unfreeze backbone). Uses AnyRes + spatial merge.

NPROC=${NPROC:-2}
MODEL_PATH=${MODEL_PATH:-"out/pointer_warmup"}   # warmup checkpoint
VISION_TOWER=${VISION_TOWER:-"mobileclip_l_1024"}
# Use GUI-Actor style YAML config listing multiple datasets
DATA_CONFIG=${DATA_CONFIG:-"/home/sdan/workspace/GUI-Actor/data/data_config.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"out/pointer_sft"}

TORCH_CMD=(python)
if [[ "$NPROC" -gt 1 ]]; then
  TORCH_CMD=(torchrun --nproc_per_node="$NPROC")
fi

# Derive AnyRes grid pinpoints from the tower's tile size (suffix in VISION_TOWER like mobileclip_l_1024)
TILE_SIZE_SUFFIX=${VISION_TOWER##*_}
if [[ "$TILE_SIZE_SUFFIX" =~ ^[0-9]+$ ]]; then
  TILE_SIZE=$TILE_SIZE_SUFFIX
else
  TILE_SIZE=1024
fi
TP1=$TILE_SIZE
TP2=$((TILE_SIZE * 2))
GRID_PINPOINTS="[("$TP1","$TP1"),("$TP2","$TP1"),("$TP2","$TP2")]"

"${TORCH_CMD[@]}" -m llava.train.train_pointer \
  --deepspeed GUI-Actor/scripts/zero3.json \
  --model_name_or_path "$MODEL_PATH" \
  --vision_tower "$VISION_TOWER" \
  --data_path "$DATA_CONFIG" \
  --image_aspect_ratio anyres \
  --image_grid_pinpoints "$GRID_PINPOINTS" \
  --mm_patch_merge_type spatial \
  --bf16 True \
  --gradient_checkpointing True \
  --group_by_modality_length True \
  --per_device_train_batch_size 1 \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --pointer_loss_weight 1.0 \
  --lm_loss_weight 1.0 \
  --freeze_backbone False \
  --output_dir "$OUTPUT_DIR"

echo "SFT complete -> $OUTPUT_DIR"
