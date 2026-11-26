#!/usr/bin/env bash
# All CIFAR-10 test images (or the first N) through the RTL of cim_gseq in parallel shards: logits, class, lost results. Golden: model/resnet_int.py
#   run_net2_all.sh [N=10000] [processes=16]
set -euo pipefail
N="${1:-10000}"; J="${2:-16}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/APlus_Project/planenest/hw/xheep_env.sh" >/dev/null
PKG="$HOME/APlus_Project/planenest/hw/x-heep/hw/core-v-mini-mcu/include/xheep_obi_pkg.sv"
OBJ="$ROOT/sim/obj_net2_all"; OUT="$ROOT/sim/net2_all_vec"; rm -rf "$OUT"; mkdir -p "$OUT"
RTL=("$ROOT/rtl/cim_engine.sv" "$ROOT/rtl/cim_fe.sv" "$ROOT/rtl/mfcc_ded.sv" "$ROOT/rtl/cim_log2fix16.sv" "$ROOT/rtl/cim_seq.sv" "$ROOT/rtl/pdm_dec.sv" "$ROOT/rtl/cim_audio.sv" "$ROOT/rtl/cim_gseq.sv" "$ROOT/rtl/cim_endpoint.sv")
verilator --binary --timing -Wno-fatal -j 8 -O2 --top-module tb_cim_net2_all -Mdir "$OBJ" -MAKEFLAGS "OBJCACHE=" -I"$ROOT/rtl" "$PKG" "${RTL[@]}" "$ROOT/tb/tb_cim_net2_all.sv" > "$ROOT/sim/logs/build_net2_all.log" 2>&1
PY="$HOME/venvs/planenest/bin/python"
cd "$ROOT/model"
"$PY" - "$N" "$J" "$OUT" <<'PYEOF'
import sys, os
import numpy as np
from resnet_int import cifar, forward
from resnet_map import Net
N, J, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
P = Net().P
x, lab = cifar(N)
per = (N + J - 1) // J
tot = 0
for s in range(J):
    a, b = s * per, min((s + 1) * per, N)
    if a >= b: break
    d = os.path.join(OUT, f"s{s}"); os.makedirs(d, exist_ok=True)
    v = forward(x[a:b], P)
    with open(os.path.join(d, "img.hex"), "w") as f:
        for i in range(a, b):
            px = x[i].astype(np.uint8).reshape(-1, 3).astype(int)
            f.write("\n".join("%06x" % (p[0] | p[1] << 8 | p[2] << 16) for p in px) + "\n")
    with open(os.path.join(d, "logits.hex"), "w") as f:
        for r in v: f.write("\n".join("%02x" % (int(z) & 255) for z in r) + "\n")
    with open(os.path.join(d, "cls.hex"), "w") as f:
        for r in v: f.write("%02x\n" % int(np.argmax(r)))
    open(os.path.join(d, "n"), "w").write(str(b - a)); tot += b - a
print(tot, "images in", s + 1, "shards; golden logits from resnet_int.py")
PYEOF
cd "$ROOT"; rm -f sim/logs/net2_all_*.log
for s in $(seq 0 $((J - 1))); do
  [ -d "$OUT/s$s" ] || continue
  ( "$OBJ/Vtb_cim_net2_all" +VDIR=sim/net2_all_vec/s$s +NIMG=$(cat "$OUT/s$s/n") > "sim/logs/net2_all_$s.log" 2>&1 ) &
done
wait
grep -h '^ALL' sim/logs/net2_all_*.log | tee sim/logs/net2_all.log
