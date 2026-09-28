#!/usr/bin/env bash
# Phase 1 (W1) — arms A and C on a 4-layer prefix.  GPU stage.
#
# Gate (docs/TAIL-EXPERIMENT-PLAN.md step 1): full-vocab KLD p99.9 and max must
# improve materially vs arm A.  PPL alone is not the signal.
#
#   arm A : --top-logits 50            the v1 reference
#   arm C : --top-logits 512           isolates top-k expansion from AYOT
#
# Arm B (AYOT traces) is NOT here: it needs teacher traces, which need the full
# BF16 teacher on a rental pod.
#
# Stop rules honoured: cache and train share --windows/--corpus-chars/--seq/--seed
# (or the KD targets desync), and every stage echoes its exact flags.
set -euo pipefail

PY="${PY:-$HOME/Desktop/work/.venv-rocm/bin/python}"
export MOE_ARTIFACTS="${MOE_ARTIFACTS:-$HOME/Desktop/work/hivebench/artifacts/ternary/moe}"
MOE="$MOE_ARTIFACTS/qwen35"
PROXY=moe/qwen35_moe_proxy.py
KLD=moe/kld_eval.py

# shared between cache and train -- must not drift
SHARED=(--prefix-layers 4 --windows 4096 --corpus-chars 50000000 --seq 512 --seed 0)

# The frozen v1 recipe, prefix-scoped.  KEEP separate from RECIPE: the KD
# weight/temp are *training* loss knobs and kld_eval.py has no such flags, so
# sharing one array makes every kld stage die on argparse.
RECIPE=(--quant lloyd --branch-quant g128 --branch-target both --rank 512
        --kd-weight 1.0 --temp 2.0)

# the subset kld_eval.py actually accepts
RECIPE_EVAL=(--quant lloyd --branch-quant g128 --branch-target both --rank 512)

# CARD selects the GPU. Default 1 (the headless card). Set CARD=0 to use the
# display card instead.
#
# One job at a TIME, not one job per card.  load_prefix materialises the prefix
# in fp32 on the host (~17 GB for a 4-layer prefix) before moving it to the GPU,
# so two concurrent load_prefix jobs exceed this box's 30 GB of RAM no matter how
# many cards are idle.  That is what killed the first --kd-weight sweep: two
# copies of the same arm plus a KLD run, all racing on one checkpoint path.
CARD="${CARD:-1}"
LOCKDIR="${LOCKDIR:-$MOE/.stage-lock}"

# One stage at a time. mkdir is atomic, so this is a real lock, and a stale lock
# from a killed run is reclaimable via STALE_PID=1 rather than blocking forever.
#
# This guard exists because the alternative was silent: two copies of the same arm
# ran concurrently, both wrote ...-step4096-kdw5.0.pt, and the machine OOM'd before
# either finished, leaving a checkpoint that was neither one's.
acquire() {
  local tag="${1:-general}" holder=""
  if ! mkdir "$LOCKDIR" 2>/dev/null; then
    holder="$(cat "$LOCKDIR/pid" 2>/dev/null || echo '?')"
    if [ -n "${STALE_PID:-}" ] && ! kill -0 "$holder" 2>/dev/null; then
      echo "reclaiming stale lock from dead pid $holder" >&2
      rm -rf "$LOCKDIR"
    else
      echo "REFUSING: stage '$holder' already holds $LOCKDIR" >&2
      echo "  another stage is running; they all load the FP prefix (~17 GB host" >&2
      echo "  RAM each), so two at once exceed this box's 30 GB." >&2
      echo "  If that process is gone, re-run with STALE_PID=1." >&2
      exit 3
    fi
    mkdir "$LOCKDIR" || { echo "REFUSING: cannot take $LOCKDIR" >&2; exit 3; }
  fi
  echo $$ > "$LOCKDIR/pid"
  echo "$tag" > "$LOCKDIR/tag"
  # release on any exit, including a signal
  trap 'rm -rf "$LOCKDIR" 2>/dev/null' EXIT INT TERM
}

run() { echo "### CARD=$CARD"; echo "### $*"; HIP_VISIBLE_DEVICES=$CARD "$@"; }

mkdir -p "$MOE" logs

case "${1:-all}" in
ref|cache|train|train-lmonly|train-kdw|kld|kld-body|steer)
  # every stage loads the FP prefix, so all of them are exclusive
  acquire "${1:-all}"
  ;;
esac

case "${1:-all}" in
ref)
  run "$PY" $PROXY ref "${SHARED[@]}" --device cuda:0 \
      --ref-file "$MOE/eval-ref-w2.pt"
  ;;

cache)
  arm="${2:?arm: a|c}"
  k=50; [ "$arm" = c ] && k=512
  run "$PY" $PROXY cache "${SHARED[@]}" --device cuda:0 \
      --top-logits $k --cache-file "$MOE/prefix-top$k.pt"
  ;;

train)
  arm="${2:?arm: a|c}"
  k=50; [ "$arm" = c ] && k=512
  # --tag is mandatory: without it both arms write the same
  # qwen35-corr-r512-g128-stepNNNN.pt and the second silently destroys the first.
  run "$PY" $PROXY train "${SHARED[@]}" "${RECIPE[@]}" \
      --device-map cuda:0 --device cuda:0 \
      --cache-file "$MOE/prefix-top$k.pt" --ref-file "$MOE/eval-ref-w2.pt" \
      --tag "arm$arm" \
      --epochs 1 --steps 4096 --eval-every 1000 --ckpt-every 1000 --log-every 100
  ;;

train-lmonly)
  # Loss-term ablation: top-512 cache, same recipe, --kd-weight 0 so the LM term
  # is the only thing training.  Answers the question the Phase 1 KLD result
  # could not: is the entropy collapse (10.77 -> 7.42 nats) caused by the LM
  # term pushing toward one-hot data targets, or by the top-k KD term?  If it
  # still sharpens with KD off, the LM term is the culprit and a tail constraint
  # bolted onto the KD term would barely move it.
  #   usage: phase1-w1.sh train-lmonly [steps]
  steps="${2:-4096}"
  run "$PY" $PROXY train "${SHARED[@]}" --quant lloyd --branch-quant g128 \
      --branch-target both --rank 512 --kd-weight 0.0 --temp 2.0 \
      --device-map cuda:0 --device cuda:0 --log-entropy \
      --cache-file "$MOE/prefix-top512.pt" --ref-file "$MOE/eval-ref-w2.pt" \
      --tag "lmonly" \
      --epochs 1 --steps "$steps" --eval-every 1000 --ckpt-every 1000 --log-every 100
  ;;

train-kdw)
  # Loss rebalance sweep, the experiment the ablation points at: --kd-weight 0
  # tripled mean KLD (1.33 -> 4.54) and sharpened harder (entropy 7.87 -> 5.70),
  # so the LM term is what collapses the tail.  This pushes the other way.
  #   usage: phase1-w1.sh train-kdw <weight> [steps]
  # Always tagged with the weight, so a sweep cannot overwrite itself.
  w="${2:?weight}"
  steps="${3:-4096}"
  # --log-entropy makes the sweep self-diagnosing: entropy and peak top-1 mass are
  # exactly what the KLD gate showed collapsing, so the shape of the answer is
  # visible in the log without a KLD run per checkpoint.
  run "$PY" $PROXY train "${SHARED[@]}" --quant lloyd --branch-quant g128 \
      --branch-target both --rank 512 --kd-weight "$w" --temp 2.0 \
      --device-map cuda:0 --device cuda:0 --log-entropy \
      --cache-file "$MOE/prefix-top512.pt" --ref-file "$MOE/eval-ref-w2.pt" \
      --tag "kdw$w" \
      --epochs 1 --steps "$steps" --eval-every 1000 --ckpt-every 1000 --log-every 100
  ;;

kld)
  # the gate instrument: same prefix, same placement, checkpoint under test.
  # NB kld_eval takes its own (narrower) flag set -- only the prefix depth and
  # the sequence length are shared with the train stage; --windows/--corpus-chars
  # are training-corpus knobs and have no meaning here.
  #   usage: phase1-w1.sh kld <ckpt> <tag> [measure-topk]
  ckpt="${2:?checkpoint}"
  tag="${3:?tag}"
  topk="${4:-0}"
  run "$PY" $KLD --prefix-layers 4 --seq 512 "${RECIPE_EVAL[@]}" --device cuda:0 \
      --eval-windows 8 --measure-topk "$topk" --load "$ckpt" --out "$MOE/kld-$tag.json"
  ;;

kld-body)
  # tail-plan Step 0 baseline: the *uncorrected* body's KLD, never measured --
  # only its PPL was.  No --load, so this is the bare ternarised body, which is
  # the honest "before" number the gate should be read against.
  run "$PY" $KLD --prefix-layers 4 --seq 512 "${RECIPE_EVAL[@]}" --device cuda:0 \
      --eval-windows 8 --out "$MOE/kld-body.json"
  ;;

steer)
  # W3 diagnostic, rides along on the same card
  run env PYTHONPATH="$HOME/Desktop/work/autogrid" \
      "$PY" moe/steer_probe.py --prefix-layers 4 --device cuda:0 \
      --windows 2 --seq 512 --quantizer lloyd --group 128 \
      --out "$MOE/steer-rank.json"
  ;;

*)
  echo "stages: ref | cache <a|c> | train <a|c> | train-lmonly [steps] |"
  echo "        train-kdw <weight> [steps] | kld <ckpt> <tag> | kld-body | steer"
  exit 2;;
esac
