#!/usr/bin/env bash
set -euo pipefail

# Pointer warmup (LM loss off, backbone frozen). Uses AnyRes + spatial merge.

# Enable CUDA debugging for device-side asserts (set to 1 for detailed error location)
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

NPROC=${NPROC:-2}                       # default for 2x A100 80GB
MODEL_PATH=${MODEL_PATH:-"checkpoints/llava-fastvithd_7b_stage3"}
VISION_TOWER=${VISION_TOWER:-"mobileclip_l_1024"}
# Use GUI-Actor style YAML config listing multiple datasets
DATA_CONFIG=${DATA_CONFIG:-"GUI-Actor/data/data_config.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"out/pointer_warmup"}
MODEL_MAX_LEN=${MODEL_MAX_LEN:-2048}
NUM_WORKERS=${NUM_WORKERS:-2}
# Resume controls:
# - If RESUME_FROM points to a checkpoint dir (…/checkpoint-XXXX), we set OUTPUT_DIR to its parent.
# - Otherwise, if START_FRESH=true, write to a new timestamped OUTPUT_DIR to avoid auto-resume.
# - Else, auto-resume if checkpoints exist under OUTPUT_DIR (handled in train_pointer.py), and we log the latest.
RESUME_FROM=${RESUME_FROM:-}
START_FRESH=${START_FRESH:-false}

TORCH_CMD=(python)
if [[ "$NPROC" -gt 1 ]]; then
  TORCH_CMD=(torchrun --nproc_per_node="$NPROC")
fi

# Derive AnyRes grid pinpoints from the tower's tile size (suffix in VISION_TOWER like mobileclip_l_1024)
# Allow override via GRID_PINPOINTS env var to control memory usage.
if [[ -z "${GRID_PINPOINTS:-}" ]]; then
  TILE_SIZE_SUFFIX=${VISION_TOWER##*_}
  if [[ "$TILE_SIZE_SUFFIX" =~ ^[0-9]+$ ]]; then
    TILE_SIZE=$TILE_SIZE_SUFFIX
  else
    # Sensible default for mobileclip_l_* families
    TILE_SIZE=1024
  fi
  # Use integer multiples of tile size to ensure whole-tile grids.
  # To reduce memory, default to two scales instead of three: [(T,T),(2T,T)]
  # You can override by exporting GRID_PINPOINTS before running this script.
  TP1=$TILE_SIZE
  TP2=$((TILE_SIZE * 2))
  GRID_PINPOINTS="[("$TP1","$TP1"),("$TP2","$TP1")]"
fi

# Handle resume/fresh logic
if [[ -n "$RESUME_FROM" ]]; then
  # Normalize path and validate
  if [[ -d "$RESUME_FROM" ]]; then
    echo "[resume] Using checkpoint dir: $RESUME_FROM"
    OUTPUT_DIR=$(dirname "$RESUME_FROM")
  else
    echo "[resume] ERROR: RESUME_FROM does not exist or is not a directory: $RESUME_FROM" >&2
    exit 1
  fi
elif [[ "$START_FRESH" == "true" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  echo "[fresh] START_FRESH=true -> writing to new output dir suffix: _$ts"
  OUTPUT_DIR="${OUTPUT_DIR}_${ts}"
else
  # Auto-detect and log latest checkpoint in OUTPUT_DIR if present
  latest_ckpt=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)
  if [[ -n "$latest_ckpt" ]]; then
    echo "[resume] Auto-detected latest checkpoint under OUTPUT_DIR: $latest_ckpt"
  fi
fi

echo "[config] NPROC=$NPROC MODEL_PATH=$MODEL_PATH"
echo "[config] OUTPUT_DIR=$OUTPUT_DIR MODEL_MAX_LEN=$MODEL_MAX_LEN"
echo "[config] GRID_PINPOINTS=$GRID_PINPOINTS VISION_TOWER=$VISION_TOWER"

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
  --model_max_length "$MODEL_MAX_LEN" \
  --dataloader_num_workers "$NUM_WORKERS" \
  --dataloader_pin_memory False \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --pointer_loss_weight 1.0 \
  --lm_loss_weight 0.0 \
  --freeze_backbone True \
  --output_dir "$OUTPUT_DIR"

echo "Warmup complete -> $OUTPUT_DIR"
