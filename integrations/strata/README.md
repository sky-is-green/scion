# Strata integration (optional)

Everything under this directory talks to the **Strata** engine. None of it is part of the Scion
model definition, and no core Scion test imports it.

| Path | What it is |
|---|---|
| `serving/` | scripts that start/benchmark the local Strata server, plus the `vram_reserve.py` sizing helper and `strata_ppl.py` (teacher-forced PPL through Strata's log-probability path). |
| `decision-heads/` | the JEV-readout scripts that use Strata's `emb=` / `/v1/verdict` endpoints. Their data and logs stay in `../../decision-heads-work/`. |
| `docs/P4-STRATA-FIT.md` | the serving-fit report for the 176B on the 2×7900 XT box. |

The Strata engine itself is a separate checkout (`sky-is-green/Strata`). The Scion repo never
vendors it and never imports it from `moe/`.

## Running the scripts

The scripts expect the workspace layout used on this box:

```
work/
  scion/          this repo
  strata/         the Strata checkout (engine + serve/server.py)
  strata-data/    MTP and expert-profile data
  models/         model files and packs
  hivebench/      Hive (configs, benchmarks)
```

Set `SCION_WORKSPACE` to override the root, `SCION_MODELS` to override the model directory, and
`SCION_REPO` to point at this checkout from elsewhere. The older session docs under `docs/` may
show the pre-move paths (`scion/serving/...`); they are historical notes, not current entry
points.

## Strata configs

The JSON engine configs (`strata-q6k-176b.json`, `strata-q2-176b-ab-rebuilt.json`, the 35B
variants) are **Hive deployment inputs**, not Scion source. They live with Hive now; the copies
in the Strata checkout are kept only until the move is confirmed.
