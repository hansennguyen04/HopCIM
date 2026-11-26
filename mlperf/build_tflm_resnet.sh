#!/usr/bin/env bash
# Build the second network's CPU-only baseline firmware (sw/applications/tflm_resnet: TensorFlow Lite Micro running MLPerf
# Tiny's ResNet8 on X-HEEP's CV32E20, docs/NET2_GATES.md N6) for the 256 KiB bitstreams at 15.36 MHz, with the same flags as
# the KWS baseline (mlperf/build_tflm.sh, which explains them).
#   usage: mlperf/build_tflm_resnet.sh <X-HEEP tree made with: xheep/setup.sh <dest> configs/kws_nexys_256k_c180.py>
# Result: <tree>/../tflm_resnet_nexys_15m36.elf (OUT=<dir> to put it elsewhere, FORCE=1 to overwrite); SIM=1: the Verilator
# SoC build, <tree>/../tflm_resnet_sim.hex.
# The configuration differs from configs/kws_nexys_256k.py only in its linker sections (code 180 KiB, data 76 KiB: patch 0008);
# the generated hardware is the same, so xheep_nexys_bscan_cim128knet2.bit (and ..._cim128ks256k.bit) run the firmware.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TREE=${1:?usage: mlperf/build_tflm_resnet.sh <X-HEEP tree made with configs/kws_nexys_256k_c180.py>}
TREE=$(realpath "$TREE")
IP=$TREE/hw/ip/hopcim
. "$HERE/../xheep/lib.sh"
[ "$(tree_cfg "$TREE")" = "$XHEEP_CFG_256K_C180" ] || die "$TREE was made with $(tree_cfg "$TREE"), not $XHEEP_CFG_256K_C180"
FLAGS="-DREFERENCE_CLOCK_Hz=15360000 -fno-rtti -fno-exceptions -fno-threadsafe-statics -fno-use-cxa-atexit \
-fno-unwind-tables -fno-asynchronous-unwind-tables -fdata-sections -DTF_LITE_STATIC_MEMORY -DNDEBUG \
-DTF_LITE_STRIP_ERROR_STRINGS -DTFLM_REMOVE_VIRTUAL_DELETE -DTF_LITE_DISABLE_X86_NEON \
-DFLATBUFFERS_LOCALE_INDEPENDENT=0 -DFLATBUFFERS_PREFER_PRINTF=1 -DTF_LITE_MCU_DEBUG_LOG \
-DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS -DTF_LITE_USE_GLOBAL_MIN -DTF_LITE_USE_GLOBAL_MAX -fpermissive \
-fno-delete-null-pointer-checks -I$IP/third_party/tflm \
-I$IP/third_party/tflm/third_party/flatbuffers/include \
-DOPENTITAN_SW_DEVICE_LIB_BASE_FREESTANDING_ASSERT_H_ -include $IP/third_party/tflm_xheep_assert.h"
if [ "${SIM:-0}" = 1 ]; then
    # the same firmware for the Verilator SoC (TARGET=sim, X-HEEP's simulation clock and UART settings, no 15.36-MHz define):
    # <out>/tflm_resnet_sim.hex, run with sim/bin/Vtestharness_cim_net2 +firmware=<hex>
    out=${OUT:-$(dirname "$TREE")}
    cd "$TREE"; xheep_tools
    exec 9> "$TREE/.hopcim-build.lock"; flock -n 9 || die "another build is running in $TREE"
    make app PROJECT=tflm_resnet TARGET=sim COMPILER_FLAGS="${FLAGS/-DREFERENCE_CLOCK_Hz=15360000 /}" > "$out/app_tflm_resnet_sim.log" 2>&1 \
        || { tail -20 "$out/app_tflm_resnet_sim.log"; die "make app failed ($out/app_tflm_resnet_sim.log)"; }
    cp sw/build/main.hex "$out/tflm_resnet_sim.hex"; cp sw/build/main.elf "$out/tflm_resnet_sim.elf"
    echo "build: $out/tflm_resnet_sim.hex  sha256 $(sha256sum "$out/tflm_resnet_sim.hex" | cut -c1-16)"
    exit 0
fi
exec "$HERE/../xheep/build.sh" "$TREE" app tflm_resnet_nexys_15m36 tflm_resnet "$FLAGS"
