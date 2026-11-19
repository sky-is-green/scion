# Runpod runbook for the 35B v2 training run

Operational notes for driving the pod from this box with the REST API v2.
Nothing here is specific to the model; the training recipe is `moe/box-run-v2.sh`.

- Base URL: `https://api.runpod.io/v2`
- Auth header: `Authorization: Bearer $RUNPOD_API_KEY`
- OpenAPI: `curl -s https://api.runpod.io/v2/openapi.json`
- The key is never printed or written to a file; pass it as an environment variable.

## Before creating anything

```sh
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/v2/account | jq .
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/v2/account/limits | jq .
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/v2/billing/balance | jq .
```

- Pods are billed **per second**; the default **spend limit is $80/hour across
  resources**, not a total budget. Check `/account/limits` before launching.
- On-demand needs at least one hour of credits for the chosen configuration.
- A pod with a network volume is stopped at $0 balance and the volume is kept;
  a pod without one is terminated and its data is lost.
- SSH keys: `GET/PUT /v2/account/ssh-keys`. `startSsh: true` injects the
  registered keys. Without a registered key there is no SSH; the `exec`
  endpoint still works.

## Choosing the GPU and data center

```sh
# availability and price for one pod GPU, secure cloud
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1" \
  | jq '.gpus[] | select(.memory>=80) | {id,memory,price,availability,dataCenters}'

curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://api.runpod.io/v2/catalog/datacenters?include=GPU_AVAILABILITY" | jq '.dataCenters[0]'
```

- Take the exact `id` from the catalog for `gpu.id`; do not type a display name.
- Target: **A100 80 GB** (cheapest that fits the 69.4 GiB bf16 body with no code
  change). H100 80 GB only if A100 has no capacity.
- Community Cloud is closed to new hosts; use `cloud: "SECURE"`.
- Network volumes are **Secure Cloud only**, and the S3 API only exists in the
  data centers listed in the storage docs (EU-CZ-1, EU-RO-1, EUR-IS-1,
  EUR-NO-1, US-CA-2, US-GA-2, US-IL-1, US-KS-2, US-MD-1, US-MO-1, US-MO-2,
  US-NC-1, US-NC-2, US-NE-1, US-WA-1). Pick one that also has the GPU.

## Storage

Prefer a **volume disk** unless you need the volume to outlive a terminate: it
has no data-center or cloud-tier constraint, so it does not narrow the A100
search. Artifacts come back with `runpodctl send/receive`, so no S3 key is
needed either.

Network volume (portable, S3-accessible, $0.07/GB/month, first 1 TB, Secure
Cloud only):

```sh
curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"scion-v2","size":150,"dataCenter":"US-KS-2"}' \
  https://api.runpod.io/v2/network-volumes | jq .
```

Volume disk instead (simpler, no data-center constraint, $0.10/GB/month while
running, $0.20 while stopped, lost on terminate): use
`"mounts": {"persistent": {"size": 150, "path": "/workspace"}}` and skip the
network volume. Artifacts then need `scp` over direct SSH.

150 GB holds the 69 GB bf16 body, the ~10 GB cache, checkpoints and the 11 GB
merged export with room to spare.

## Creating the pod

```sh
curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{
    "name": "scion-v2-train",
    "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "cloud": "SECURE",
    "gpu": {"id": "GPU_ID_FROM_CATALOG", "count": 1, "minRamPerGpu": 64, "minVcpuCountPerGpu": 8},
    "dataCenterIds": ["US-KS-2"],
    "disk": 50,
    "mounts": {"network": [{"volumeId": "VOLUME_ID", "path": "/workspace"}]},
    "ports": ["22/tcp"],
    "startSsh": true,
    "env": {
      "HF_HOME": "/workspace/hf",
      "HF_HUB_ENABLE_HF_TRANSFER": "1",
      "MOE_ARTIFACTS": "/workspace/artifacts",
      "REPO": "/workspace/scion",
      "SCION_COMMIT": "scion-test"
    }
  }' https://api.runpod.io/v2/pods | jq .
```

`image` is required unless `templateId` is set. `minRamPerGpu` is a placement
filter, not an allocation; 64 GB keeps the host from being a 32 GB box, which
matters for the 69 GB download plus page cache.

Poll until `status` is `RUNNING`:

```sh
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.io/v2/pods/$POD_ID | jq '{status,ssh,dataCenterId,cost}'
```

`ssh.direct` is populated only when `22/tcp` is published; it is the one that
supports `scp`. `ssh.proxy` is interactive-only.

## Running commands without SSH

`POST /pods/{id}/exec` runs an argv array inside the container and returns
`stdout`, `stderr`, `exitCode`, `timedOut`, `durationMs`. **The timeout is 25 s
maximum**, so anything longer is launched detached and polled:

```sh
curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"command":["bash","-lc","cd /workspace/scion && nohup bash moe/box-run-v2.sh setup > /workspace/setup.log 2>&1 & echo started"],"timeoutSeconds":10}' \
  https://api.runpod.io/v2/pods/$POD_ID/exec | jq -r .stdout

# poll
curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"command":["bash","-lc","tail -n 40 /workspace/setup.log"]}' \
  https://api.runpod.io/v2/pods/$POD_ID/exec | jq -r .stdout
```

`GET /pods/{id}/logs` gives the container stdout; use it for the runner's own
output.

## Stage map

`setup → smoke → curriculum → cache → ref → train (cur05) → eval → export`

- `setup` clones `sky-is-green/scion` at `SCION_COMMIT` and
  `sky-is-green/scion-llama.cpp` (gguf-py), installs the pinned deps and
  downloads `empero-ai/Qwen3.8-35B-A3B-Distill` (public, ~69 GB).
- `curriculum` scores the 4096 windows and writes `curric-combo.jsonl`; it is
  a required stage now, not an upload.
- `train-fallback` (pred2.0) is opt-in; run it only after cur05's eval.
- The 200-step pilot is `train` with `--steps 200` (or the stage's env
  override) and is the gate for the full run.

## Syncing artifacts

Simplest path, no extra credentials, works on either storage type. On the pod:

```sh
runpodctl send /workspace/artifacts/qwen35/qwen35-adapter-cur05-final.gguf
# prints a code phrase
```

Then on this box:

```sh
runpodctl receive <code>
```

`send`/`receive` ride Runpod's croc relay, so they also work without an exposed
port or SSH key. Pull the adapters, eval JSONs, training logs and
`curric-combo.jsonl`; leave the 11 GB merged body on the volume unless needed.

With a network volume instead, from this box (needs the separate Runpod S3 key
pair):

```sh
export AWS_ACCESS_KEY_ID=...      # user_... from the S3 API key
export AWS_SECRET_ACCESS_KEY=...  # rps_...
aws s3 cp --region US-KS-2 --endpoint-url https://s3api-us-ks-2.runpod.io/ \
  s3://VOLUME_ID/qwen35/qwen35-adapter-cur05-final.gguf .
```

Without one, `scp -P <direct port> root@<ip>:/workspace/... .` over
`ssh.direct`.

Download at minimum: the adapter GGUFs, the eval JSONs, the training logs,
and `curric-combo.jsonl`. The merged 11 GB body can stay on the volume.

## Stopping and cost

```sh
curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"action":"stop"}' https://api.runpod.io/v2/pods/$POD_ID/action
curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.io/v2/pods/$POD_ID
```

- `stop` keeps `/workspace` on a volume disk but bills it at $0.20/GB/month;
  on a network volume it bills $0.07/GB/month and survives termination.
- `start` can come back with **zero GPUs** if the host's GPUs were taken;
  restart is safer but keeps the machine.
- Delete the volume once the artifacts are synced. 150 GB at $0.07 is about
  $10.50/month, which will quietly outlive the run if forgotten.

## Agent tooling (not needed for this run)

`runpodctl` already covers GPU/DC availability, pods, volumes, logs and file
transfer; only `exec` needs REST. The skills/MCP are generic onboarding and are
not required here. If they are ever wanted for opencode:

```sh
npx -y skills add runpod/runpod-plugins-official --skill '*' --agent opencode --yes --global
npx @runpod/mcp-server@latest add --client opencode
# then, in opencode:  /mcp  -> runpod -> sign in
```

With no interactive sign-in, register the server with the key instead of OAuth:

```json
{"mcp": {"runpod": {"type": "remote", "url": "https://mcp.runpod.io/",
  "headers": {"Authorization": "Bearer ${RUNPOD_API_KEY}"}}}}
```

The REST API above is sufficient for this run; the skills are convenience.
