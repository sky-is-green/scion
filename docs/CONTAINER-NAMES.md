# Scion container names

The `P` in the old Scion names was a branding moniker, not a format distinction.  The canonical
display names are now:

| Canonical | Legacy alias | Type id | Block | Layout |
|---|---|---:|---:|---|
| `TQ1_0_g128` | `PTQ1_0` | 143 | 128 values / 28 B | TQ1_0 base-3 packing at group size 128 |
| `Q2_0_g128` | `PQ2_0` | 142 | 128 values / 34 B | Q2_0's 2-bit codec at group size 128 |
| `Q1_0_g128` | — | 43 | 128 values / 34 B | correction container, same block shape as `Q2_0_g128` |

These are **not** the standard g256 containers:

| Standard | Type id | Block |
|---|---:|---:|
| `TQ1_0` | 34 | 256 values / 54 B |
| `TQ2_0` | 35 | 256 values / 66 B |
| `Q2_0` | 42 | 64 values / 18 B |
| `TQ2_T` | 144 | 128 values / 34 B, trellis-coded |

So `TQ1_0_g128` and `Q2_0_g128` keep the `_g128` suffix on purpose.  `Q2_0_g128` must not be
called `TQ2_0`: it is the Q2_0 codec, not the standalone TQ2_0 container.  `PQ2_0` had the same
block size as `TQ2_T` but a different bitstream, so the two must not be aliased either.

## Compatibility

- GGUF type ids and block geometry are unchanged.  Existing model files keep loading as before.
- `PTQ1_0` and `PQ2_0` remain accepted as **input aliases** in tools and logs.
- New code, logs and user-facing text should use `TQ1_0_g128` and `Q2_0_g128`.
- Artifact filenames such as `Scion-35B-A3B-PQ2_0-corr.gguf` are historical names and are not
  renamed by this convention.

## Internal symbols

The C++/Python symbols `PTQ1_0`, `PQ2_0`, `ptq1_0.py`, `block_ptq1_0` and `block_pq2_0` stay for
ABI and source compatibility.  They refer to the canonical containers above; the display layer is
where the naming change lives.
