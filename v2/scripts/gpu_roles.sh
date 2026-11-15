#!/bin/bash
# gpu_roles.sh - the canonical card roles for this box.  Source it; do not
# hardcode card indices anywhere else.
#
# THE POLICY (2026-11-15): when both cards are used, the DISPLAY card always
# carries the lesser strain - less VRAM, less compute.  The SPARE card runs
# near capacity.  Concretely: the spare is strata's primary (device 0); the
# display is the peer/expert tier, capped by --peer-slots, with its desktop
# allowance in --peer-reserve-mib.
#
# Numbering schemes on this box (do not trust any single one alone):
#   sysfs:   card0 = spare, card1 = display (detected: connected outputs)
#   strata:  device 0 = sysfs card1 (display), device 1 = sysfs card0 (spare)
#   HIP:     default order puts card1 first, so HIP_VISIBLE_DEVICES="1,0"
#            swaps it: HIP device 0 = spare, HIP device 1 = display.
#
# Usage:  source gpu_roles.sh
#         then use $SYSFS_DISPLAY / $SYSFS_SPARE for sysfs telemetry, and
#         $HIP_ORDER_SPARE_FIRST as HIP_VISIBLE_DEVICES for engines.

detect_display_card() {
    for c in /sys/class/drm/card*; do
        local n; n=$(basename "$c")
        [ -e "$c/device" ] || continue
        for s in "$c"-"DP-"*/status "$c"-"HDMI-"*/status; do
            [ -e "$s" ] || continue
            if [ "$(cat "$s")" = "connected" ]; then
                echo "$n"; return
            fi
        done
    done
    echo ""
}

SYSFS_DISPLAY=${SYSFS_DISPLAY:-$(detect_display_card)}
SYSFS_SPARE=${SYSFS_SPARE:-$(cd /sys/class/drm && ls -d card* | grep -v "^$SYSFS_DISPLAY$" | head -1)}

if [ -z "$SYSFS_DISPLAY" ] || [ "$SYSFS_DISPLAY" != "card1" ]; then
    echo "gpu_roles: WARNING the detected display card ($SYSFS_DISPLAY) does not" \
         "match this box's known mapping (card1) - check the HIP order before" \
         "launching engines" >&2
fi

# strata's device 0 is sysfs card1 on this box (whatever is HIP-first today),
# so spare-first ordering swaps the two:
HIP_ORDER_SPARE_FIRST="1,0"

# telemetry helper:  gpu_roles stat "card0"  -> "used MiB busy %"
gpu_roles_stat() {
    local d=$1
    local used busy
    used=$(cat /sys/class/drm/$d/device/mem_info_vram_used 2>/dev/null)
    busy=$(cat /sys/class/drm/$d/device/gpu_busy_percent 2>/dev/null)
    echo "$d: $((used / 1024 / 1024)) MiB busy ${busy:-?}%"
}
