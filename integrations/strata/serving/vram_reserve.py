#!/usr/bin/env python3
"""vram_reserve.py - size the Strata engine's VRAM headroom for this machine.

The engine's `--vram-reserve-mib` is one number applied to every card, so the binding
constraint is the card that carries the desktop: its free VRAM shrinks when the compositor,
a browser or another agent starts using the GPU, and a card working at the fringe of its
memory slows down (and can fail later allocations). Measured on 2026-10-13, this box
(2x RX 7900 XT 20 GiB, display on card1): with the engine's cache filled to the edge the
display card showed 428 MiB free and decode ran ~25.8 tok/s; with a reserve that keeps
>= 2.2 GiB free it measured 27-33 tok/s.

Rule:
  * a card with a connected display keeps  max(2048 MiB, 10% of total)  free;
  * any other card keeps               max(1024 MiB,  5% of total)  free;
  * add 1024 MiB for the buffers the engine allocates AFTER the expert-cache fit
    (prompt path, verify windows, hand-off rings). Measured: a 2560 MiB reserve left
    1781 MiB free on the display card after the desktop grew during load, so the fit
    number is not the final free number.
  * the flag is global, so the recommendation is the largest per-card need; a
    non-display card simply gets the same headroom.

Usage:
  python3 vram_reserve.py                # card table + recommended --vram-reserve-mib
  python3 vram_reserve.py --apply CONF   # write the recommendation into a serve config
  python3 vram_reserve.py --check        # exit 1 when a display card is below target
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

DRM = pathlib.Path("/sys/class/drm")


def cards() -> list[dict]:
    out = []
    for dev in sorted(DRM.glob("card[0-9]*/device")):
        name = dev.parent.name  # card0, card1, ...
        try:
            used = int((dev / "mem_info_vram_used").read_text())
            total = int((dev / "mem_info_vram_total").read_text())
        except OSError:
            continue
        connected = any(
            (c / "status").read_text().strip() == "connected"
            for c in DRM.glob(f"{name}-*")
            if (c / "status").exists()
        )
        out.append({"card": name, "used_mib": used // 1048576, "total_mib": total // 1048576,
                    "free_mib": (total - used) // 1048576, "display": connected})
    return out


def headroom_target(card: dict) -> int:
    total = card["total_mib"]
    if card["display"]:
        return max(2048, total // 10)
    return max(1024, total // 20)


def recommend(rows: list[dict]) -> int:
    target = max(headroom_target(c) for c in rows) if rows else 2048
    return target + 1024  # engine buffers after the fit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", metavar="CONFIG", help="serve config JSON to update")
    ap.add_argument("--check", action="store_true", help="exit 1 when a display card is below target")
    args = ap.parse_args()

    rows = cards()
    if not rows:
        print("no amdgpu cards found under /sys/class/drm", file=sys.stderr)
        return 2

    worst = None
    print("card   total    used     free   display   target_free")
    for c in rows:
        t = headroom_target(c)
        flag = "!" if c["display"] and c["free_mib"] < t else ""
        print(f"{c['card']:6s} {c['total_mib']:7d} {c['used_mib']:7d} {c['free_mib']:8d} "
              f"{'yes' if c['display'] else 'no':>7s}   {t:5d}  {flag}")
        if c["display"] and (worst is None or c["free_mib"] < worst):
            worst = c["free_mib"]

    rec = recommend(rows)
    print(f"\nrecommended --vram-reserve-mib {rec}  "
          f"(display headroom {max(headroom_target(c) for c in rows)} + 1024 engine buffers)")

    if args.check:
        for c in rows:
            if c["display"] and c["free_mib"] < headroom_target(c):
                print(f"WARNING: {c['card']} (display) has {c['free_mib']} MiB free, "
                      f"below target {headroom_target(c)} MiB", file=sys.stderr)
                return 1

    if args.apply:
        path = pathlib.Path(args.apply)
        cfg = json.loads(path.read_text())
        a = cfg["args"]
        if "--vram-reserve-mib" in a:
            a[a.index("--vram-reserve-mib") + 1] = str(rec)
        else:
            a += ["--vram-reserve-mib", str(rec)]
        path.write_text(json.dumps(cfg, indent=1) + "\n")
        print(f"updated {path}: --vram-reserve-mib {rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
