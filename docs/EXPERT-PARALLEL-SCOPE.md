# Expert parallelism across the two cards — scoping (2026-10-13)

**Why now.** After the correction reshape (`P4-STRATA-FIT.md` item 5) the
with/without-corrections arms sit at ~41.4 vs ~46.4 tok/s on the same engine
build. The corrections now cost 2.6 ms/token; the two arms converged onto a
shared ceiling, and the ceiling is not the kernels: measured during a 400-token,
temperature-0 decode,

| counter (1 s average, `/sys/class/drm/cardX/device/gpu_busy_percent`) | card0 | card1 |
| --- | ---: | ---: |
| SM busy | **48.9%** | **54.8%** |
| memory-controller busy | 10.8% | 5.7% |

The shares are the layer split's arithmetic (26/48 and 22/48 of the layers), and
that is structural: the verifier's stage chain is serial - a stage runs its own
layers, writes the residual hand-off, then calls the next stage (`verify.cpp`,
the `le_ < g.n_layers` branch). For a single conversation one card computes
while the other waits, every window. Memory bandwidth is idle; the currency is
SM time.

**Ceiling.** A decode window is ~65 ms with ~31 ms of stamped GPU stage time;
each card is busy ~half of that. If both cards did useful work during the same
layer, the ~31 ms of per-layer stage time could compress toward its
half-card-serial chains, and the HarMoEny result (arXiv 2506.12417, +37-70%
under imbalance) bounds what scheduling alone can recover. The hand-off budget
is small and known: a token's activation is 10 KB, the k selected experts'
outputs are ~10 KB each (k=10: ~100 KB/token/layer; at 40 tok/s over 48 layers
that is ~0.2 GB/s against ~25 GB/s of PCIe), so this is not a bandwidth problem
- it is dispatch, ownership and scheduling.

## What expert parallelism would change

Today each card's expert cache holds the experts of **its own layers**; every
expert the router picks for layer l is on l's owner. The profile-filled cache
reaches 99.8% hits because the hot pairs repeat within the layer.

EP re-partitions by (expert, owner) **across all layers**: each card holds a
slice of every layer's expert set and computes its slice while the other card
computes the rest of the same layer. One token's layer becomes:

1. the router runs where the activation is (unchanged);
2. for each of the k selected experts, the activation goes to that expert's
   owner card (a mapped copy; both caches hold only the owners' slices);
3. each owner computes its experts on the window's rows (`native_expert_grouped`
   already batches by expert);
4. the expert outputs come back and are combined (`moe_combine`), as today.

The dependencies inside the MoE block are per-expert, so the two cards' work is
independent until the combine; the combine then awaits both. The 1 ms-scale
stages on the critical path are the copies and the combine, not the experts.

## The offline go/no-go: done, and it clears

The measured routing is on disk: `moe/artifacts/qwen4exp/router-stats-coder.json`
(2.4M tokens, per-layer per-expert counts, k=10) and the live cache sizes
(card0 12007 slots for layers 0-25, card1 10690 for 26-47, 92% of all pairs).
`moe/ep_sim.py` fills each card's slots with the hottest pairs of whatever
ownership layout is asked and reports the routed mass that stays resident plus
the per-card load:

| layout | card0 load | card1 load | card0 resident | card1 resident |
| --- | ---: | ---: | ---: | ---: |
| current (card = layer owner) | 55.3% | 44.7% | 99.40% | 99.99% |
| EP, snake split per layer by count | 50.5% | 49.5% | 100.00% | 99.44% |
| EP, global expert id by total mass | 50.1% | 49.9% | 100.00% | 99.42% |

**EP is free on the cache** (overall resident 99.7% in every layout - the
capacity spends itself on the same hot pairs) and it fixes the load split
(55.3/44.7 today, from the 26/22 layer split -> 50/50).  The per-token split of
the k=10 picks is not in the aggregate counts; a routing trace is needed for its
tail, and that is the one open question the corpus cannot answer offline
(the marginal approximation is in the script's output, not relied on here).

## The scoping questions, in order

1. ~~Cache economics~~ **done above** - clears.
2. **Dispatch shape.** Where the selected experts for a row are split across
   owners, how many mapped copies per layer, and can they band with the
   existing doorbell/plan protocol (`verify.cpp`'s `G` groups and plan sink)
   instead of adding a second one?
3. **Skew.** HarMoEny's +37-70% is the gain under imbalance; the steady-state
   gain here is bounded by whichever card's slice is busier. The split above
   answers the expected load (50/50); the per-token tail needs the trace.
4. **The draft layer.** It already sits on the last stage; upstream 0.1.44
   moves the draft layer to the first card in a pipelined decode. Either way it
   is single-card work that could hide under the other card's verify - ~3.5
   ms/window.

## What this is not

- Not a rewrite of the expert kernels; the grouped native expert kernel and the
  cache stay as they are, only the ownership map and the dispatch change.
- Not tensor parallelism (that would change every projection; llama.cpp's
  `-sm tensor` needs RCCL and is off by default upstream - not applicable to
  our containers).
- Not the multi-GPU work upstream is doing: theirs stays pipeline (draft layer
  on the first card, 8K prompt chunks), and RDNA3 keeps 0.1.42 behavior. The
  pipeline's cards still take turns for one conversation; EP is the change that
  makes them work at the same time.

## Proposed order

1. ~~Offline go/no-go~~ **done, clears** (table above; `moe/ep_sim.py`).
2. Prototype the dispatch on the verify path behind an env flag, with the
   existing parity tests and the fixed protocol as the gate; card-busy
   ~50% -> >80% (per card, `/sys/class/drm/cardX/device/gpu_busy_percent`) is
   the measurement that says it works.  Open design point: the per-token split
   tail needs a routing trace before the dispatcher's buffering is sized.
3. File the summary upstream either way - it is the vendor's engine and they
   handle AMD; the numbers above are the case.
