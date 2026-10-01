#!/usr/bin/env bash
# Stage 2 ablation: full fine-tuning vs attention-only, from the same stage-1 checkpoint.
# Both runs share seed, data order, steps, learning rates and the fixed-mask held-out set,
# so the eval_loss curves are directly comparable.
#
#   bash scripts/compare_trainable.sh            # defaults below
#   STAGE2_STEPS=1000 BODY_LRS="5e-5 1e-4" bash scripts/compare_trainable.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-checkpoints/ablation}
STAGE1_STEPS=${STAGE1_STEPS:-5000}
STAGE2_STEPS=${STAGE2_STEPS:-3000}
BODY_LRS=${BODY_LRS:-5e-5}             # space-separated; each is run for both variants
EVAL_STEPS=${EVAL_STEPS:-250}
# 16GB V100: 4 x 1024 tokens per step, 32 sequences per optimizer update.
COMMON=${COMMON:---batch_size 4 --grad_accum 8 --seq_len 1024 --gradient_checkpointing --eval_steps $EVAL_STEPS}

if [ ! -f "$OUT/stage1/model.safetensors" ]; then
    uv run python train.py --stage 1 --max_steps "$STAGE1_STEPS" --output_dir "$OUT/stage1" $COMMON
fi

for lr in $BODY_LRS; do
    for trainable in attention all; do
        run="$OUT/stage2-$trainable-lr$lr"
        [ -f "$run/model.safetensors" ] && { echo "skip $run (done)"; continue; }
        uv run python train.py --stage 2 --init_from "$OUT/stage1" --trainable "$trainable" \
            --body_lr "$lr" --max_steps "$STAGE2_STEPS" --output_dir "$run" $COMMON
    done
done

uv run python scripts/compare_runs.py "$OUT"/stage2-*
