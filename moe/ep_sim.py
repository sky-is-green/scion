#!/usr/bin/env python3
"""ep_sim.py - expert-parallelism go/no-go from the measured routing, offline.

Inputs: router-stats-coder.json (per-layer per-expert counts, 2.4M tokens) and the
running service's cache sizes (card0 12007 slots for layers 0-25, card1 10690 for
26-47).  Compares the current layer-owner layout with two expert-parallel ownership
splits, on (a) the routed mass resident (cache hit rate) and (b) the per-card load
share (imbalance), with a per-token split distribution under the marginal counts.
"""
import json
import sys

STATS = '/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/scion/moe/artifacts/qwen4exp/router-stats-coder.json'
SLOTS0, SLOTS1 = 12007, 10690
SPLIT_LAYER = 26  # card0 = layers 0..25, card1 = 26..47

d = json.load(open(STATS))
NL, NE = d['n_layer'], d['n_expert']
counts = [[0] * NE for _ in range(NL)]
for layer in d['layers']:
    counts[layer['il']] = layer['counts']
total_all = sum(sum(c) for c in counts)
print(f"routing: {d['tokens']} tokens, {NL} layers x {NE} experts, k={d['n_expert_used']}, "
      f"total routed events {total_all}")
print(f"cache: card0 {SLOTS0} slots (layers 0..{SPLIT_LAYER-1}), card1 {SLOTS1} slots "
      f"(layers {SPLIT_LAYER}..{NL-1})\n")


def hit(card_pairs, slots):
    """card_pairs: list of (count, layer, expert); fill `slots` hottest -> routed mass covered."""
    ranked = sorted(card_pairs, reverse=True)
    covered = sum(c for c, _, _ in ranked[:slots])
    total = sum(c for c, _, _ in ranked)
    return covered, total


def report(name, cards):
    tot_cov = tot_all_ev = 0
    shares = []
    for ci, pairs in enumerate(cards):
        cov, ev = hit(pairs, SLOTS0 if ci == 0 else SLOTS1)
        tot_cov += cov
        tot_all_ev += ev
        shares.append(ev / total_all)
        print(f"  {name}: card{ci} load {100*ev/total_all:5.1f}% of routed mass; "
              f"resident with {SLOTS0 if ci == 0 else SLOTS1} slots -> {100*cov/ev:5.2f}% of its load")
    print(f"  {name}: overall resident {100*tot_cov/total_all:5.2f}% of all routed mass; "
          f"per-card load {100*shares[0]:.1f}/{100*shares[1]:.1f}")
    return shares


print("A. current layout (card = layer owner)")
cards = [[] for _ in range(2)]
for l in range(NL):
    card = 0 if l < SPLIT_LAYER else 1
    for e in range(NE):
        cards[card].append((counts[l][e], l, e))
report('current', cards)

print("\nB. EP, snake split per layer by routing count (each card 256 experts/layer)")
cards = [[] for _ in range(2)]
for l in range(NL):
    order = sorted(range(NE), key=lambda e: -counts[l][e])
    for i, e in enumerate(order):
        cards[i % 2].append((counts[l][e], l, e))
shareB = report('ep-snake', cards)

print("\nC. EP, global expert ownership by total routing mass (same expert id across layers)")
tot = [sum(counts[l][e] for l in range(NL)) for e in range(NE)]
order = sorted(range(NE), key=lambda e: -tot[e])
cards = [[] for _ in range(2)]
owner = {}
for i, e in enumerate(order):
    owner[e] = i % 2
for l in range(NL):
    for e in range(NE):
        cards[owner[e]].append((counts[l][e], l, e))
shareC = report('ep-global', cards)

# per-token split of the 10 chosen experts across cards, under the marginal counts.
# Poisson-binomial over p_e = 10 * count_e / sum(counts[l]) (an approximation: the router
# picks the top 10 of its logits, counts are the marginals).
print("\nper-token split under the marginal counts (mean over layers):")
for name, owner in (('ep-snake', None), ('ep-global', None)):
    pass
# snake owner map
ownerB = {}
for l in range(NL):
    order = sorted(range(NE), key=lambda e: -counts[l][e])
    for i, e in enumerate(order):
        ownerB[(l, e)] = i % 2
mean0, p_all0, p_all1 = 0.0, 0.0, 0.0
for l in range(NL):
    s = sum(counts[l])
    probs = [min(1.0, 10.0 * counts[l][e] / s) for e in range(NE)]
    # Poisson-binomial DP truncated at 10
    dp = [0.0] * (11)
    dp[0] = 1.0
    for p in probs:
        for j in range(10, 0, -1):
            dp[j] = dp[j] * (1 - p) + dp[j - 1] * p
        dp[0] *= (1 - p)
    mean0 += sum(j * dp[j] for j in range(11))
    p_all0 += dp[0]
    p_all1 += dp[10]
print(f"  ep-snake: mean experts on card0 {mean0/NL:.2f} of 10; "
      f"P(all 10 on card0) {100*p_all0/NL:.2f}%; P(all 10 on card1) {100*p_all1/NL:.2f}%")
