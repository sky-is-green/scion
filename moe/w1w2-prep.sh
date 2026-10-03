#!/bin/bash
# w1w2-prep.sh — stage W1+W2 exactly (local, free).  Verifies the inputs,
# writes the deployed step-3000 checkpoint, snapshots the shard-protection
# set, computes the local-export shard fetch list, and prints the exact
# booking / upload / stage / guard / export command plan.
#
#   bash scion/moe/w1w2-prep.sh            # prep + print plan
#   bash scion/moe/w1w2-prep.sh --bundle   # also build the upload tarball
#
# W1+W2 (cap $6, L40S 48 GB US-TX-4 @ $1.09/hr, 188 GB host, 400 GB disk,
# no network volume):
#   W1 = 48-layer KLD gate on the deployed step-3000 checkpoint
#        (gate-teacher + gate-student, both the proven --full dense-load path)
#   W2 = PLE precision sweep on the pod fp8 route (2-layer ple-ab)
#   export = LOCAL, after fetch (no-PLE file by design, like the reference)
set -euo pipefail

WORK="${WORK:-$HOME/Desktop/work}"
MOE="${MOE_ARTIFACTS:-$WORK/hivebench/artifacts/ternary/moe}"
Q="$MOE/qwen4exp"
P2B="$Q/pod/remote-p2b"
MD="$WORK/models/qwen38-flashnext-fp8"
SHARDS="$MD/shards"
UP="$Q/w1w2-upload"
CKPT_SRC="$P2B/qwen4exp-corr-r512-g128-step3000-cur05.pt"
CKPT_SHA="4b14f32b166ba2daffdb3199be54a80e0a79ca7e8730fe78eeb362a739ae29fb"
CKPT_DEPLOYED="$UP/qwen4exp-corr-r512-g128-step3000-cur05-deployed.pt"
KEEP="$UP/keep-shards.txt"
FETCH="$UP/fetch-shards.txt"
BUNDLE="$UP/scion-flashnext-w1w2.tar.gz"
PY="${PY:-$WORK/hivebench/.venv/bin/python}"

die() { echo "REFUSING: $*" >&2; exit 1; }

[ -f "$CKPT_SRC" ] || die "step-3000 checkpoint missing: $CKPT_SRC"
got=$(sha256sum "$CKPT_SRC" | awk '{print $1}')
[ "$got" = "$CKPT_SHA" ] || die "step-3000 sha mismatch: $got != $CKPT_SHA"
[ -d "$SHARDS" ] || die "mirror shards missing: $SHARDS"
[ -f "$MD/model.safetensors.index.json" ] || die "mirror index missing"

mkdir -p "$UP"

# 1. deployed checkpoint: branch+router only (balance_bias is training-time
#    shaping; the eval strict check and the release export both use this form)
if [ ! -f "$CKPT_DEPLOYED" ]; then
    "$PY" - "$CKPT_SRC" "$CKPT_DEPLOYED" <<'EOF'
import sys
sys.path.insert(0, "scion/moe")
import qwen4exp_export as qx
info = qx.write_deployed_checkpoint(sys.argv[1], sys.argv[2])
print(f"deployed checkpoint: {info}")
EOF
else
    echo "deployed checkpoint exists: $CKPT_DEPLOYED"
fi
sha256sum "$CKPT_DEPLOYED" | tee "$CKPT_DEPLOYED.sha256"

# 2. shard protection + the 48-layer local-export fetch list
ls "$SHARDS" > "$KEEP"
"$PY" - "$MD" "$SHARDS" "$FETCH" <<'EOF'
import json, os, sys
sys.path.insert(0, "scion/moe")
import qwen4exp_export as qx
md, shards_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
idx = json.load(open(os.path.join(md, "model.safetensors.index.json")))["weight_map"]
local = set(os.listdir(shards_dir))
# build a headless exporter just for its read-key plan
from argparse import Namespace
ex = qx.Exporter(Namespace(
    model_dir=md, layers=48, experts="ptq1_0", body="f16", branches="none",
    branch_dtype="f16", branch_quant="g128", deploy_quant="lloyd",
    routers="replace", adapter_recipe="", use_temp_file=False,
    release_shards="none", keep_shards="", out="/dev/null"))
need = set()
for il in range(48):
    for k in ex._read_keys(il):
        for kk in (k, k + "_scale_inv"):
            s = idx.get(kk)
            if s:
                need.add(s)
for hf, _, _ in qx.SHARED_JOBS:
    for kk in (hf, hf + "_scale_inv"):
        s = idx.get(kk)
        if s:
            need.add(s)
missing = sorted(need - local)
with open(out, "w") as f:
    f.write("\n".join(missing) + ("\n" if missing else ""))
print(f"48-layer export needs {len(need)} shards; "
      f"{len(missing)} missing ({len(local & need)} local)")
EOF

# 3. optional upload bundle
if [ "${1:-}" = "--bundle" ]; then
    tar --exclude=.git --exclude=__pycache__ --exclude='*.gguf' \
        -czf "$BUNDLE" -C "$WORK" scion
    sha256sum "$BUNDLE"
fi

cat <<EOF

=== W1+W2 PLAN (ask the human BEFORE booking; never book unapproved) ===

[preflight]
  balance: read-only check (needs ~/.runpod-api-key):
    source $WORK/scion/moe/pod_watch/common.sh; pw_balance
  UPDATE 2026-10-03: W1+W2 ran; **W2 done** (pod PLE sweep 4-bit 0.0015,
  sha a94a73d7), **W1 retry approved (option A) with a sub-cap \$4**.
  Balance \$17.03.  This plan now covers the W1 retry only; skip [5].
  Guard: cap \$6 -> stop target is the W1 gate JSON, reserve floor \$0.5
  (computed \$0.41) => cap-stop at \$5.5 spent, i.e. ~\$4 of fresh headroom.
  **Rebuild the bundle (--bundle) before upload: the current one predates
  the lean loader (streamed expert assignment) and the manual-CPU
  gate-teacher stage.**
  pod: RTX 6000 Ada 48 GB SECURE \$0.84/hr, minRAMPerGPU=180, minVCPUPerGPU=16,
  containerDisk 100 GB + volume 300 GB (the 173 GiB teacher lives on
  /workspace = the volume; a 100 GB volume would fail setup), ports 22/tcp,
  image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 (P2b-identical:
  torch 2.9.1+cu128, driver 570). Fallback: community 6000 Ada (needs
  supportPublicIp=true) -> L40 secure \$0.82 -> L40S secure \$1.09.
  no network volume -> setup re-downloads the teacher.
  LOCAL: suite 383 passed + 2 skipped; 4-layer load proof on dGPU1; branch
  merge parse-back; cap_eval server smoke done.
  inputs to upload: bundle + deployed ckpt; the 11.39 GB cache is NOT needed
  (W1 uses wikitext windows; W2 uses the 2-layer prefix).

[0. book]  (ONLY after the human's go + top-up; REST create; capture id/IP/port)
  curl -s -X POST https://rest.runpod.io/v1/pods \\
    -H "Authorization: Bearer \$(cat ~/.runpod-api-key)" \\
    -H 'Content-Type: application/json' -d '{
      "name": "flashnext-w1w2",
      "imageName": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
      "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1,
      "gpuTypeIds": ["NVIDIA RTX 6000 Ada Generation", "NVIDIA L40", "NVIDIA L40S"],
      "gpuTypePriority": "custom",
      "minRAMPerGPU": 180, "minVCPUPerGPU": 16,
      "containerDiskInGb": 100, "volumeInGb": 300,
      "volumeMountPath": "/workspace",
      "ports": ["22/tcp"], "interruptible": false,
      "env": {"SSH_PUBLIC_KEY": "'"\$(cat ~/.ssh/id_runpod.pub)"'",
              "PUBLIC_KEY": "'"\$(cat ~/.ssh/id_runpod.pub)"'"}
    }'
  # capture: pod id, publicIp, portMappings["22"] -> POD/IP/PORT below.
  # If create refuses (no secure stock): retry with "cloudType":"COMMUNITY"
  # + "supportPublicIp": true; the gpuTypeIds order already falls back
  # Ada -> L40 -> L40S. Never blind-terminate if ssh is down (guard polls).

[1. upload]  (from this box; set IP/PORT/POD from the booking)
  ssh -i ~/.ssh/id_runpod -p \$PORT root@\$IP 'mkdir -p /workspace/scion /workspace/artifacts/qwen4exp'
  scp -i ~/.ssh/id_runpod -P \$PORT $BUNDLE root@\$IP:/workspace/
  scp -i ~/.ssh/id_runpod -P \$PORT $CKPT_DEPLOYED root@\$IP:/workspace/artifacts/qwen4exp/
  ssh -i ~/.ssh/id_runpod -p \$PORT root@\$IP 'cd /workspace && tar -xzf scion-flashnext-w1w2.tar.gz -C /workspace/scion --strip-components=1 && sha256sum /workspace/artifacts/qwen4exp/qwen4exp-corr-r512-g128-step3000-cur05-deployed.pt'
  # remote sha must equal the local deployed sha (905af765...): verify before
  # any stage; this is the P2b "missing input" trap closed.

[2. setup + smoke]  (~20 min)
  ssh ... 'cd /workspace/scion && bash moe/box-run-flashnext.sh setup'
  ssh ... 'cd /workspace/scion && bash moe/box-run-flashnext.sh smoke'
  # setup must print native_fp8_kept=true only in preflight; smoke drift ~0.38
  # NOTE: setup deletes nothing; it downloads the teacher when absent.

[3. guards FIRST, then stages]  (watchdogs: scion/moe/pod_watch/)
  # spend guard: cap \$6, L40S rate, fetch-before-terminate; stops when the
  # W2 artifact is fetched + sha-verified
  PW_POD=\$POD PW_SSH_HOST=root@\$IP PW_SSH_PORT=\$PORT PW_DST=$Q/pod/remote-w1w2 \\
  PW_WINDOW_CAP=6 PW_RATE_PER_HR=<booked-rate> PW_FETCH_MIN=15 PW_MARGIN_MIN=10 \\
  PW_RESERVE_FLOOR=0.5 \\
  PW_CKPTS="qwen4exp-eval-kld-48l-step3000.json" \\
  PW_KILL_SIBLING="pod_watch/puller.sh" \\
  nohup bash $WORK/scion/moe/pod_watch/spend_guard.sh > $Q/pod/remote-w1w2/w1w2-spend.log 2>&1 &
  # survival puller (5-min cycles, verify on)
  PW_POD=\$POD PW_SSH_HOST=root@\$IP PW_SSH_PORT=\$PORT PW_DST=$Q/pod/remote-w1w2 \\
  PW_VERIFY=1 nohup bash $WORK/scion/moe/pod_watch/puller.sh > $Q/pod/remote-w1w2/w1w2-pull.log 2>&1 &
  # per-stage watchdog: 30-min log-idle for the dense-load stages (compact
  # prints every 4 layers; a true stall is hours -- attempt 8), 15 min for
  # ple-ab.  Gate logs have no ^step lines, so only the log guard applies.
  PW_STAGE=gate-teacher PW_SSH_HOST=root@\$IP PW_SSH_PORT=\$PORT PW_DST=$Q/pod/remote-w1w2 \\
  PW_LOG_IDLE_S=1800 PW_KILL_PATTERN="qwen4exp_eval.py" \\
  nohup bash $WORK/scion/moe/pod_watch/stage_watchdog.sh > $Q/pod/remote-w1w2/w1w2-watch-teacher.log 2>&1 &
  # (repeat the watchdog with PW_STAGE=gate-student PW_LOG_IDLE_S=1800, then
  #  PW_STAGE=ple-ab PW_LOG_IDLE_S=900 PW_KILL_PATTERN="qwen4exp_proxy.py ple-ab")

[4. W1 gate]  (~1-1.5 h: manual CPU teacher load+pass, then native student)
  # teacher = manual bf16 on CPU (native fp8 forward is impossible on 48 GB;
  # the lean loader streams expert banks so the 286 GB cgroup holds)
  ssh ... 'cd /workspace/scion && bash moe/pod_watch/stage-exec.sh gate-teacher2 bash moe/box-run-flashnext.sh gate-teacher'
  # wait for gate-teacher2.rc = 0, then FETCH THE PARK (the guard does not
  # include tcache-48l in its patterns; without this a pod loss redoes 1 h):
  rsync -a --partial -e 'ssh -i ~/.ssh/id_runpod -p \$PORT' \
    root@\$IP:/workspace/artifacts/qwen4exp/tcache-48l/ $Q/pod/remote-w1w2/tcache-48l/
  # then the student (native compact path, P2b-proven):
  ssh ... 'cd /workspace/scion && bash moe/pod_watch/stage-exec.sh gate-student bash moe/box-run-flashnext.sh gate-student'
  # artifacts: qwen4exp-eval-kld-48l-step3000.json (teacher entropy + KLD
  # mean/p99/max, top1, decomposition) + logs

[5. W2 PLE sweep]  **DONE 2026-10-03** (4-bit 0.0015, sha a94a73d7) — skip.

[stall recovery]  (only if a stage watchdog exits 2)
  # TERM trap clears the stage lock; belt-and-braces:
  ssh ... 'rm -rf /workspace/artifacts/.stage-lock'
  # relaunch ONLY the failed stage; a finished teacher park survives a
  # student failure (gate-student reads $Q/tcache-48l)

[6. close]  spend guard fetches + terminates on the W1 gate JSON (or cap/reserve)
  # if it is still up after the artifacts land, terminate via the guard or:
  #   PW_POD=\$POD PW_SSH_HOST=root@\$IP PW_SSH_PORT=\$PORT PW_DST=... \\
  #   PW_WINDOW_CAP=6 bash .../spend_guard.sh   # (same stop path)
  # verify locally:
  sha256sum $Q/pod/remote-w1w2/qwen4exp-eval-*.json

[7. LOCAL export hook]  (after the window; ~25-35 min, \$0)
  # 7a. fetch the missing mirror shards (list written by this prep):
  while read -r f; do
    [ -s $SHARDS/\$f ] && continue
    curl -sL -C - --retry 5 --retry-delay 5 -o $SHARDS/\$f.part \\
      "https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/resolve/main/\$f" \\
      && mv $SHARDS/\$f.part $SHARDS/\$f
  done < $FETCH
  # 7b. export the deployed step-3000 arm with the embedded adapter:
  mkdir -p $WORK/tmp-export
  TMPDIR=$WORK/tmp-export "$PY" $WORK/scion/moe/qwen4exp_export.py \\
    --model-dir $MD --layers 48 --experts ptq1_0 --body f16 \\
    --use-temp-file --release-shards all \\
    --keep-shards "\$(paste -sd, $KEEP)" \\
    --branches $CKPT_DEPLOYED --branch-dtype f16 --routers replace \\
    --adapter-recipe "Flash-Next cur05 step3000 r512 g128 lloyd, deployed" \\
    --out $Q/qwen4exp-48l-ptq1_0-corr-step3000.gguf
  # 7c. parse-back + record:
  "$PY" $WORK/scion/moe/export_check.py $Q/qwen4exp-48l-ptq1_0-corr-step3000.gguf
  sha256sum $Q/qwen4exp-48l-ptq1_0-corr-step3000.gguf

=== end plan ===
EOF
