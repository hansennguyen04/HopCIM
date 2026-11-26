// HopCIM's, not vendored: what assert.h means for the CPU-only TFLM baseline (sw/applications/tflm_mlperf),
// included first in every C and C++ file of that build (mlperf/build_tflm.sh: -include).
//
// X-HEEP puts its own freestanding assert.h on the include path ahead of the C library's. That file ignores
// NDEBUG and, in C++, calls a function (_writestr) it does not declare, so TFLM and FlatBuffers do not compile
// against it, and with it their asserts would run. The build defines that file's include guard
// (OPENTITAN_SW_DEVICE_LIB_BASE_FREESTANDING_ASSERT_H_), so it adds nothing, and this file supplies what the
// two headers stand for in this build:
//   assert         nothing, as the C library's assert.h with NDEBUG (TFLM's release build sets NDEBUG);
//   static_assert  C11's _Static_assert in C, as both headers define it (C++ has the keyword).
// The assembler files of the build (crt0.S) see none of it.
#ifndef TFLM_XHEEP_ASSERT_H_
#define TFLM_XHEEP_ASSERT_H_
#ifndef __ASSEMBLER__

#ifndef NDEBUG
#error "the TFLM baseline is a release build: NDEBUG must be defined"
#endif
#undef assert
#define assert(expression) ((void)0)
#if !defined(__cplusplus) && !defined(static_assert)
#define static_assert _Static_assert
#endif

#endif  // __ASSEMBLER__
#endif  // TFLM_XHEEP_ASSERT_H_
