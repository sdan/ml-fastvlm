#!/usr/bin/env bash
set -euo pipefail

# Pointer warmup (LM loss off, backbone frozen). Uses AnyRes + spatial merge.

NPROC=${NPROC:-2}                       # default for 2x A100 80GB
MODEL_PATH=${MODEL_PATH:-"checkpoints/llava-fastvithd_7b_stage3"}
VISION_TOWER=${VISION_TOWER:-"mobileclip_l_1024"}
# Use GUI-Actor style YAML config listing multiple datasets
# Default to a local config that points at /home/sdan/workspace/GUI-Actor/GUI-Actor-Data
DATA_CONFIG=${DATA_CONFIG:-"scripts/data_config_local.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"out/pointer_warmup"}

TORCH_CMD=(python)
if [[ "$NPROC" -gt 1 ]]; then
  TORCH_CMD=(torchrun --nproc_per_node="$NPROC")
fi

# Derive AnyRes grid pinpoints from the tower's tile size (suffix in VISION_TOWER like mobileclip_l_1024)
TILE_SIZE_SUFFIX=${VISION_TOWER##*_}
if [[ "$TILE_SIZE_SUFFIX" =~ ^[0-9]+$ ]]; then
  TILE_SIZE=$TILE_SIZE_SUFFIX
else
  # Sensible default for mobileclip_l_* families
  TILE_SIZE=1024
fi
# Use integer multiples of tile size to ensure whole-tile grids and handle portrait/landscape:
# [(T,T), (2T,T), (T,2T), (2T,2T)]
TP1=$TILE_SIZE
TP2=$((TILE_SIZE * 2))
GRID_PINPOINTS="[("$TP1","$TP1"),("$TP2","$TP1"),("$TP1","$TP2"),("$TP2","$TP2")]"

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
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --pointer_loss_weight 1.0 \
  --lm_loss_weight 0.0 \
  --freeze_backbone True \
  --output_dir "$OUTPUT_DIR"

echo "Warmup complete -> $OUTPUT_DIR"
