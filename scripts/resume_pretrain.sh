#!/bin/bash
#
# Resume pretraining from a saved checkpoint.
#
# 1. To STOP current training: Press Ctrl+C in the terminal running pretrain.py
#
# 2. To RESUME:
#    - From latest (most recent epoch, use if you just stopped):
#      ./scripts/resume_pretrain.sh latest
#
#    - From best (best accuracy so far):
#      ./scripts/resume_pretrain.sh best
#
# Customize the variables below for your setup.

SOURCE_DOMAIN=${SOURCE_DOMAIN:-ucf}
DATA_ROOT=${DATA_ROOT:-./data/ucf_hmdb_full}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints}
NUM_CLASSES=${NUM_CLASSES:-12}

case "${1:-latest}" in
  latest)
    RESUME_CHECKPOINT="${OUTPUT_DIR}/source_${SOURCE_DOMAIN}_latest.pth"
    ;;
  best)
    RESUME_CHECKPOINT="${OUTPUT_DIR}/source_${SOURCE_DOMAIN}_best.pth"
    ;;
  *)
    RESUME_CHECKPOINT="$1"
    ;;
esac

if [ ! -f "$RESUME_CHECKPOINT" ]; then
  echo "Error: Checkpoint not found: $RESUME_CHECKPOINT"
  echo "Run pretrain.py first to create checkpoints."
  exit 1
fi

echo "Resuming from: $RESUME_CHECKPOINT"
echo ""

python scripts/pretrain.py \
  --config config/default.yaml \
  --source_domain "$SOURCE_DOMAIN" \
  --data_root "$DATA_ROOT" \
  --num_classes "$NUM_CLASSES" \
  --epochs 30 \
  --batch_size 8 \
  --lr 0.01 \
  --weight_decay 0.0001 \
  --early_stop_patience 5 \
  --i3d_rgb_pretrained pretrained/rgb_imagenet.pt \
  --i3d_flow_pretrained pretrained/flow_imagenet.pt \
  --output_dir "$OUTPUT_DIR" \
  --resume "$RESUME_CHECKPOINT"
