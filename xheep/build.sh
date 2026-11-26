#!/usr/bin/env bash
# Build HopCIM's FPGA bitstreams, Verilator SoC models and firmware in an X-HEEP tree made by
# xheep/setup.sh. One build at a time per tree (a lock enforces it): X-HEEP's boot-ROM generator
# runs in place, and a second FuseSoC run in the same tree deletes the first one's image.
#
# usage: xheep/build.sh <tree> <variant>                 a bitstream or a Verilator model (list below)
#        xheep/build.sh <tree> app <name> <project> [<compiler flags>]   firmware for the Nexys A7
# env:   OUT    where results go (default <tree>/../bitstreams for bitstreams, <tree>/.. otherwise);
#               an existing result is not overwritten unless FORCE=1
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# variant | kind | build root | FuseSoC flags | result name | what it is
VARIANTS='
s     | fpga | build_micks | use_bscane_xilinx use_cim use_cim_mic use_cim_s              | xheep_nexys_bscan_cim128ks.bit    | baseline S, live audio, 15.36 MHz
s4    | fpga | build_mic4s | use_bscane_xilinx use_cim use_cim_mic use_cim_s use_cim_mic4 | xheep_nexys_bscan_cim128mic4s.bit | baseline S at its lowest real-time clock, 4.096 MHz
b1    | fpga | build_mickb1 | use_bscane_xilinx use_cim use_cim_mic use_cim_s use_cim_b1  | xheep_nexys_bscan_cim128kb1.bit   | S with StreamFuse B in block 1, live audio, 15.36 MHz
b1-4  | fpga | build_mic4b1 | use_bscane_xilinx use_cim use_cim_mic use_cim_s use_cim_mic4 use_cim_b1 | xheep_nexys_bscan_cim128mic4b1.bit | B1 at 4.096 MHz
dwu   | fpga | build_mickdwu | use_bscane_xilinx use_cim use_cim_mic use_cim_s use_cim_dwu | xheep_nexys_bscan_cim128kdwu.bit  | S with the digital depthwise unit (docs/DWU_GATES.md), live audio, 15.36 MHz
dwu-4 | fpga | build_mic4dwu | use_bscane_xilinx use_cim use_cim_mic use_cim_s use_cim_mic4 use_cim_dwu | xheep_nexys_bscan_cim128mic4dwu.bit | the same at 4.096 MHz
s256  | fpga | build_micks256 | use_bscane_xilinx use_cim use_cim_mic use_cim_s           | xheep_nexys_bscan_cim128ks256k.bit | baseline S with 256 KiB of RAM (tree from setup.sh <dest> configs/kws_nexys_256k.py)
r6b   | fpga | build_mick  | use_bscane_xilinx use_cim use_cim_mic                        | xheep_nexys_bscan_cim128k.bit     | R6b sequencer, live audio, 15.36 MHz
r6b5  | fpga | build_mic5  | use_bscane_xilinx use_cim use_cim_mic use_cim_mic5           | xheep_nexys_bscan_cim128mic5.bit  | R6b at its lowest real-time clock, 5.12 MHz
net2-256 | fpga | build_micks_net2 | use_bscane_xilinx use_cim use_cim_mic use_cim_s use_cim_dwu use_cim_net2 | xheep_nexys_bscan_cim128knet2.bit | S with the depthwise unit and the second network (ResNet8, docs/NET2_GATES.md), live audio, 15.36 MHz, 256 KiB of RAM (tree from setup.sh <dest> configs/kws_nexys_256k.py)
sim   | sim  | -           | use_cim                                                      | Vtestharness_cim                  | Verilator SoC, R6b, CIM front-end
sim-ded | sim | -          | use_cim use_cim_ded                                          | Vtestharness_ded                  | Verilator SoC, R6b, dedicated front-end B3
sim-s | sim  | -           | use_cim use_cim_s                                            | Vtestharness_cim_s                | Verilator SoC, baseline S
sim-b1 | sim | -           | use_cim use_cim_s use_cim_b1                                 | Vtestharness_cim_b1               | Verilator SoC, S with B in block 1
sim-dwu | sim | -          | use_cim use_cim_s use_cim_dwu                                | Vtestharness_cim_dwu              | Verilator SoC, S with the digital depthwise unit
sim-net2-256 | sim | -    | use_cim use_cim_s use_cim_dwu use_cim_net2                    | Vtestharness_cim_net2             | Verilator SoC, S with the unit and the second network, 256 KiB
'
usage() { echo "usage: xheep/build.sh <tree> <variant> | <tree> app <name> <project> [<compiler flags>]"; echo "variants:"; echo "$VARIANTS" | sed '/^$/d; s/|[^|]*|[^|]*|[^|]*|[^|]*|/:/; s/^/  /'; exit 1; }

TREE=$(realpath "${1:-}" 2> /dev/null) || usage
V=${2:-}
[ -n "$V" ] || usage
[ -f "$TREE/core-v-mini-mcu.core" ] || die "$TREE is not an X-HEEP tree"
cd "$TREE"
xheep_tools
exec 9> "$TREE/.hopcim-build.lock"
flock -n 9 || die "another build is running in $TREE"

save() {  # source, destination
    if [ -e "$2" ] && [ "${FORCE:-0}" != 1 ]; then die "$2 exists (FORCE=1 to overwrite)"; fi
    cp "$1" "$2"
    echo "build: $2  sha256 $(sha256sum "$2" | cut -c1-16)"
}

if [ "$V" = app ]; then
    name=${3:?app needs <name> <project>}; project=${4:?app needs <name> <project>}; flags=${5:-}
    out=${OUT:-$(dirname "$TREE")}
    echo "build: app $name for a tree made with $(tree_cfg "$TREE")"
    make app PROJECT="$project" TARGET=nexys-a7-100t COMPILER_FLAGS="$flags" > "$out/app_$name.log" 2>&1 \
        || { tail -20 "$out/app_$name.log"; die "make app failed ($out/app_$name.log)"; }
    save sw/build/main.elf "$out/$name.elf"
    exit 0
fi

row=$(echo "$VARIANTS" | awk -F'|' -v v="$V" '{k = $1; gsub(/ /, "", k)} k == v')   # not gsub on $1: that rebuilds $0 without the |
[ -n "$row" ] || usage
kind=$(echo "$row" | cut -d'|' -f2 | xargs); root=$(echo "$row" | cut -d'|' -f3 | xargs)
flags=$(echo "$row" | cut -d'|' -f4 | xargs); name=$(echo "$row" | cut -d'|' -f5 | xargs)
fflags=$(for f in $flags; do printf -- '--flag=%s ' "$f"; done)
# the generator configuration the variant needs (the RTL and the memory map come from it)
case $V in *256) want=$XHEEP_CFG_256K;; *) want=$XHEEP_CFG;; esac
[ "$(tree_cfg "$TREE")" = "$want" ] || die "$V needs a tree made with $want (xheep/setup.sh <dest> $want); $TREE has $(tree_cfg "$TREE")"
t0=$SECONDS
if [ "$kind" = fpga ]; then
    out=${OUT:-$(dirname "$TREE")/bitstreams}
    mkdir -p "$out"
    [ ! -e "$out/$name" ] || [ "${FORCE:-0}" = 1 ] || die "$out/$name exists (FORCE=1 to overwrite)"
    echo "build: $V -> $name ($flags), build root $root/core, log $TREE/$root.build.log"
    # --build-root keeps FuseSoC's default depth (the board files are found as ../../../)
    fusesoc --cores-root . run --no-export --target=nexys-a7-100t --build-root "$root/core" $fflags \
        --build openhwgroup.org:systems:core-v-mini-mcu > "$root.build.log" 2>&1 \
        || { tail -20 "$root.build.log"; die "build failed ($TREE/$root.build.log)"; }
    d="$root/core/nexys-a7-100t-vivado/openhwgroup.org_systems_core-v-mini-mcu_1.0.6.runs/impl_1"
    save "$d/xilinx_core_v_mini_mcu_wrapper.bit" "$out/$name"
    cp "$d/xilinx_core_v_mini_mcu_wrapper_timing_summary_routed.rpt" "$out/${name%.bit}.timing.rpt"
    cp "$d/xilinx_core_v_mini_mcu_wrapper_utilization_placed.rpt" "$out/${name%.bit}.utilization.rpt" 2> /dev/null || true
else
    out=${OUT:-$(dirname "$TREE")}
    echo "build: $V -> $name ($flags), log $TREE/$V.build.log"
    make verilator-build FUSESOC_FLAGS="$fflags" > "$V.build.log" 2>&1 \
        || { tail -20 "$V.build.log"; die "build failed ($TREE/$V.build.log)"; }
    save build/openhwgroup.org_systems_core-v-mini-mcu_1.0.6/sim-verilator/Vtestharness "$out/$name"
fi
echo "build: $V done in $((SECONDS - t0)) s"
