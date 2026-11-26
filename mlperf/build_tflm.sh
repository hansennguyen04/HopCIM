#!/usr/bin/env bash
# Build the CPU-only baseline firmware (sw/applications/tflm_mlperf: TensorFlow Lite Micro running MLPerf
# Tiny's KWS model on X-HEEP's CV32E20) for the 256 KiB S bitstream at 15.36 MHz.
#   usage: mlperf/build_tflm.sh <X-HEEP tree made with: xheep/setup.sh <dest> configs/kws_nexys_256k.py>
# Result: <tree>/../tflm_mlperf_nexys_15m36.elf (OUT=<dir> to put it elsewhere, FORCE=1 to overwrite).
# Flags: TFLM's own for a release build (BUILD_TYPE=release: NDEBUG, error strings stripped; static memory,
# no RTTI or exceptions; its Makefile's common defines) and its RISC-V target's defines and C++ flags
# (tensorflow/lite/micro/tools/make/targets/riscv32_generic_makefile.inc), at X-HEEP's -O2 like every HopCIM
# firmware; -I for TFLM's headers (third_party/tflm,
# reached through the tree's hw/ip/hopcim link; absolute, since CMake also compiles a test file elsewhere). X-HEEP puts its own
# freestanding assert.h on the include path; it ignores NDEBUG (TFLM's release build relies on assert compiling to
# nothing) and does not compile as C++, so it is kept empty by defining its include guard, and
# third_party/tflm_xheep_assert.h, included first, gives assert the C library's NDEBUG meaning (X-HEEP's drivers
# do not call assert; uart.c's static_assert keeps its meaning).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TREE=${1:?usage: mlperf/build_tflm.sh <X-HEEP tree made with configs/kws_nexys_256k.py>}
TREE=$(realpath "$TREE")
IP=$TREE/hw/ip/hopcim
. "$HERE/../xheep/lib.sh"
[ "$(tree_cfg "$TREE")" = "$XHEEP_CFG_256K" ] || die "$TREE was made with $(tree_cfg "$TREE"), not $XHEEP_CFG_256K"
FLAGS="-DREFERENCE_CLOCK_Hz=15360000 -fno-rtti -fno-exceptions -fno-threadsafe-statics -fno-use-cxa-atexit \
-fno-unwind-tables -fno-asynchronous-unwind-tables -fdata-sections -DTF_LITE_STATIC_MEMORY -DNDEBUG \
-DTF_LITE_STRIP_ERROR_STRINGS -DTFLM_REMOVE_VIRTUAL_DELETE -DTF_LITE_DISABLE_X86_NEON \
-DFLATBUFFERS_LOCALE_INDEPENDENT=0 -DFLATBUFFERS_PREFER_PRINTF=1 -DTF_LITE_MCU_DEBUG_LOG \
-DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS -DTF_LITE_USE_GLOBAL_MIN -DTF_LITE_USE_GLOBAL_MAX -fpermissive \
-fno-delete-null-pointer-checks -I$IP/third_party/tflm \
-I$IP/third_party/tflm/third_party/flatbuffers/include \
-DOPENTITAN_SW_DEVICE_LIB_BASE_FREESTANDING_ASSERT_H_ -include $IP/third_party/tflm_xheep_assert.h"
exec "$HERE/../xheep/build.sh" "$TREE" app tflm_mlperf_nexys_15m36 tflm_mlperf "$FLAGS"
