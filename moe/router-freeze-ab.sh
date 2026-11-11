#!/usr/bin/env bash
# Fixed-router vs joint-router correction A/B on the OLMoE proxy.
#
# Two arms, one cache, one seed, one step budget:
#   joint  : correction branches + trainable routers (the current recipe)
#   frozen : correction branches only (--freeze-router)
#
# The training recipe is the deployable one: Lloyd g128 body, g128 sidecar,
# rank-512 branches on both placements, LR decay from half-time, router-KD off
# (D4).  Evaluate with the routing/load diagnostics added to the eval stage.
#
# Usage:
#   bash moe/router-freeze-ab.sh                 # 4096 steps, seed 0
#   STEPS=1024 EVAL_WINDOWS=8 bash moe/router-freeze-ab.sh
#   # one cache per training seed: the cache is built from seed-specific
#   # windows, so a different --seed needs a different teacher cache.
#   CACHE=$MOE_ARTIFACTS/olmoe/teacher-cache-s1.pt SEEDS=1 \
#       bash moe/router-freeze-ab.sh
#
# It assumes the teacher cache and eval refs already exist:
#   $MOE_ARTIFACTS/olmoe/teacher-cache.pt
#   $MOE_ARTIFACTS/olmoe/eval-ref-w2.pt
# Build them with moe/README.md steps 1 and 1b.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ART=${MOE_ARTIFACTS:-"$REPO/moe/artifacts"}
CACHE=${CACHE:-"$ART/olmoe/teacher-cache.pt"}
REF=${REF:-"$ART/olmoe/eval-ref-w2.pt"}
OUT="$ART/olmoe"
DEVICE=${DEVICE:-cuda:0}
STEPS=${STEPS:-4096}
WINDOWS=${WINDOWS:-4096}
CORPUS=${CORPUS:-50000000}
EVAL_WINDOWS=${EVAL_WINDOWS:-8}
RANK=${RANK:-512}
SEED=${SEED:-0}
SEEDS=${SEEDS:-"$SEED"}
TAG=${TAG:-rf}
QUANT=${QUANT:-lloyd}
BRANCH_QUANT=${BRANCH_QUANT:-g128}
TARGET=${TARGET:-both}
LR=${LR:-1e-3}
ROUTER_LR=${ROUTER_LR:-0.0}

# The teacher cache is built from seed-specific text windows; a cache built
# for seed 0 cannot supervise a seed-1 student.  Keep the invariant explicit
# rather than silently producing a misaligned run.
if [ "$(wc -w <<<"$SEEDS")" -gt 1 ]; then
    echo "error: SEEDS needs one cache per seed." >&2
    echo "Build a cache with --seed N, then run with CACHE=... SEEDS=N." >&2
    exit 2
fi

[ -f "$CACHE" ] || { echo "missing cache: $CACHE" >&2; exit 2; }
[ -f "$REF" ] || { echo "missing eval refs: $REF" >&2; exit 2; }

run_arm () {
    local seed=$1 arm=$2 extra=$3
    local tag="${TAG}-s${seed}-${arm}"
    echo "== train $tag (steps $STEPS, rank $RANK, seed $seed)"
    # shellcheck disable=SC2086
    python "$REPO/moe/olmoe_corrections.py" train \
        --device-map "$DEVICE" --device "$DEVICE" \
        --cache-file "$CACHE" --ref-file "$REF" \
        --quant "$QUANT" --branch-quant "$BRANCH_QUANT" --branch-target "$TARGET" \
        --rank "$RANK" --seed "$seed" \
        --windows "$WINDOWS" --corpus-chars "$CORPUS" \
        --epochs 1 --steps "$STEPS" \
        --lr "$LR" --router-lr "$ROUTER_LR" \
        --lr-half-every 500 --lr-decay-start 2000 \
        --router-weight 0 \
        --tag "$tag" $extra

    local ckpt
    ckpt=$(ls -t "$OUT"/olmoe-corr-r"$RANK"-*"$tag"*.pt 2>/dev/null | head -1)
    [ -n "$ckpt" ] || { echo "no checkpoint for $tag" >&2; exit 3; }

    echo "== eval $tag (trained routers)"
    python "$REPO/moe/olmoe_corrections.py" eval \
        --device "$DEVICE" --device-map "$DEVICE" \
        --quant "$QUANT" --branch-quant "$BRANCH_QUANT" --branch-target "$TARGET" \
        --rank "$RANK" --seed "$seed" --eval-windows "$EVAL_WINDOWS" \
        --load "$ckpt" --tag "${tag}-trained" \
        $extra

    if [ "$arm" = "joint" ]; then
        echo "== eval $tag (base routers: branch-only ablation)"
        python "$REPO/moe/olmoe_corrections.py" eval \
            --device "$DEVICE" --device-map "$DEVICE" \
            --quant "$QUANT" --branch-quant "$BRANCH_QUANT" --branch-target "$TARGET" \
            --rank "$RANK" --seed "$seed" --eval-windows "$EVAL_WINDOWS" \
            --load "$ckpt" --tag "${tag}-base" --eval-router base \
            $extra
    fi
}

for seed in $SEEDS; do
    run_arm "$seed" joint ""
    run_arm "$seed" frozen "--freeze-router"
done

echo "== done"
ls -1 "$OUT"/branches-eval-"$TAG"-*.json 2>/dev/null || true
