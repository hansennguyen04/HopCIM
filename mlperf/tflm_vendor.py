#!/usr/bin/env python3
"""Vendor the part of TensorFlow Lite Micro that MLPerf Tiny's KWS model needs into the CPU-only baseline
firmware (sw/applications/tflm_mlperf), and the part ResNet8 needs into the second network's CPU-only baseline
(sw/applications/tflm_resnet, docs/NET2_GATES.md N6), and write both models as C arrays.

TFLM is taken at a pinned commit: the one the tflite-micro Python package that made mlperf/reference_kws01.json
was built from (0.dev20261112231828 = commit c0ecf6ff, 2026-11-12 23:18:28 UTC), so the board and the
reference run the same kernels. Steps:
  1. TFLM's own project generator (create_tflm_tree.py) makes a standalone tree from the checkout;
  2. the sources the firmware links (SOURCES: the interpreter, the allocator and the kernels of the
     reference's six operators: found by linking every TFLM source with --gc-sections and keeping the
     files that contribute code) are copied to <app>/tflm, and every header they or tflm_glue.cpp
     include (g++ -MM, X-HEEP's RISC-V GCC) to third_party/tflm, with TFLM's and FlatBuffers' licences.
     ResNet8's firmware (SOURCES_RESNET) links the same files without the depthwise kernel and with the
     ADD kernel; third_party/tflm holds the headers of both;
     The headers stay outside sw/: X-HEEP's build puts every directory under an app that holds a header
     on the include path, and FlatBuffers' flatbuffers/string.h would then hide <string.h>. The X-HEEP
     tree reaches them through its hw/ip/hopcim link (mlperf/build_tflm.sh);
  3. third_party/tflm/VENDORED.md records the commit, the generator command and the sha256 of every file;
  4. <app>/kws_model_data.cpp holds kws_ref_model.tflite (MLPerf Tiny's KWS model, unchanged) byte for
     byte, 16-byte aligned, with its sha256; tflm_resnet/resnet_model_data.cpp likewise
     pretrainedResnet_quant.tflite (MLPerf Tiny's image-classification model).

  python3 mlperf/tflm_vendor.py <tflite-micro checkout at c0ecf6ff> <kws_ref_model.tflite> <pretrainedResnet_quant.tflite>
  (needs riscv64-unknown-elf-g++ on PATH: source planenest/hw/xheep_env.sh)
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "sw", "applications", "tflm_mlperf")
APP_RESNET = os.path.join(HERE, "..", "sw", "applications", "tflm_resnet")
INC = os.path.join(HERE, "..", "third_party", "tflm")
COMMIT = "c0ecf6ffaa571068fa905339776161124d4b77dd"
MODEL_SHA256 = "aeea436800704fce"          # first 16 hex digits of kws_ref_model.tflite's sha256
RESNET_SHA256 = "3c002613d1b2475e"         # ... of pretrainedResnet_quant.tflite's
SOURCES = """
tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cpp
tensorflow/lite/micro/flatbuffer_conversions.cpp
tensorflow/lite/micro/flatbuffer_utils.cpp
tensorflow/lite/micro/kernels/conv.cpp
tensorflow/lite/micro/kernels/conv_common.cpp
tensorflow/lite/micro/kernels/depthwise_conv.cpp
tensorflow/lite/micro/kernels/depthwise_conv_common.cpp
tensorflow/lite/micro/kernels/fully_connected.cpp
tensorflow/lite/micro/kernels/fully_connected_common.cpp
tensorflow/lite/micro/kernels/internal/common.cpp
tensorflow/lite/micro/kernels/internal/portable_tensor_utils.cpp
tensorflow/lite/micro/kernels/internal/quantization_util.cpp
tensorflow/lite/micro/kernels/kernel_util.cpp
tensorflow/lite/micro/kernels/log_softmax.cpp
tensorflow/lite/micro/kernels/pooling.cpp
tensorflow/lite/micro/kernels/pooling_common.cpp
tensorflow/lite/micro/kernels/reshape.cpp
tensorflow/lite/micro/kernels/reshape_common.cpp
tensorflow/lite/micro/kernels/softmax.cpp
tensorflow/lite/micro/kernels/softmax_common.cpp
tensorflow/lite/micro/memory_helpers.cpp
tensorflow/lite/micro/memory_planner/greedy_memory_planner.cpp
tensorflow/lite/micro/memory_planner/linear_memory_planner.cpp
tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cpp
tensorflow/lite/micro/micro_allocation_info.cpp
tensorflow/lite/micro/micro_allocator.cpp
tensorflow/lite/micro/micro_common.cpp
tensorflow/lite/micro/micro_context.cpp
tensorflow/lite/micro/micro_interpreter.cpp
tensorflow/lite/micro/micro_interpreter_context.cpp
tensorflow/lite/micro/micro_interpreter_graph.cpp
tensorflow/lite/micro/micro_op_resolver.cpp
tensorflow/lite/micro/micro_resource_variable.cpp
tensorflow/lite/micro/micro_utils.cpp
""".split()
SOURCES_RESNET = sorted([s for s in SOURCES if "/depthwise_conv" not in s] +
                        ["tensorflow/lite/micro/kernels/add.cpp", "tensorflow/lite/micro/kernels/add_common.cpp"])
LICENSES = ["LICENSE", "third_party/flatbuffers/LICENSE"]
# the flags of tflm_mlperf's build (mlperf/build_tflm.sh) that change what the preprocessor includes
CPPFLAGS = ["-march=rv32imc", "-mabi=ilp32", "-std=gnu++17", "-DTF_LITE_STATIC_MEMORY", "-DNDEBUG",
            "-DTF_LITE_STRIP_ERROR_STRINGS", "-DTFLM_REMOVE_VIRTUAL_DELETE", "-DTF_LITE_DISABLE_X86_NEON",
            "-DFLATBUFFERS_LOCALE_INDEPENDENT=0", "-DFLATBUFFERS_PREFER_PRINTF=1", "-DTF_LITE_MCU_DEBUG_LOG",
            "-DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS", "-DTF_LITE_USE_GLOBAL_MIN", "-DTF_LITE_USE_GLOBAL_MAX"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def model_cpp(path, model, name, var, guard_h, what):
    b = open(model, "rb").read()
    with open(path, "w") as fh:
        fh.write(f"// {what}, byte for byte: {len(b)} bytes, sha256 {sha(model)}.\n"
                 "// Written by mlperf/tflm_vendor.py.\n"
                 f"#include \"{guard_h}\"\n\n"
                 f"alignas(16) const unsigned char {var}[] = {{\n")
        for i in range(0, len(b), 16):
            fh.write("    " + ", ".join(f"0x{x:02x}" for x in b[i:i + 16]) + ",\n")
        fh.write(f"}};\nconst unsigned int {var}_len = {len(b)};\n")
    return len(b)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    tflm, model, model_r = (os.path.abspath(a) for a in sys.argv[1:4])
    head = subprocess.run(["git", "-C", tflm, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == COMMIT, f"{tflm} is at {head}, not {COMMIT}"
    assert sha(model).startswith(MODEL_SHA256), f"{model} is not MLPerf Tiny's kws_ref_model.tflite"
    assert sha(model_r).startswith(RESNET_SHA256), f"{model_r} is not MLPerf Tiny's pretrainedResnet_quant.tflite"
    out, out_r = os.path.join(APP, "tflm"), os.path.join(APP_RESNET, "tflm")
    apps = [(APP, SOURCES), (APP_RESNET, SOURCES_RESNET)]
    with tempfile.TemporaryDirectory() as tmp:
        tree = os.path.join(tmp, "tree")
        gen = ["python3", "tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py", "--rename_cc_to_cpp", tree]
        subprocess.run(gen, cwd=tflm, check=True, capture_output=True)
        inc = ["-I", tree, "-I", os.path.join(tree, "third_party", "flatbuffers", "include")]
        headers = set()
        for app, srcs in apps:
            for src in [os.path.join(tree, s) for s in srcs] + [os.path.join(app, "tflm_glue.cpp")]:
                dep = subprocess.run(["riscv64-unknown-elf-g++", *CPPFLAGS, *inc, "-I", app, "-MM", src],
                                     capture_output=True, text=True, check=True).stdout
                for tok in dep.replace("\\\n", " ").split()[1:]:
                    p = os.path.abspath(tok)
                    if p.startswith(tree + os.sep) and p.endswith(".h"):
                        headers.add(os.path.relpath(p, tree))
        for d in (out, out_r, INC):
            if os.path.exists(d):
                shutil.rmtree(d)
        placed = ([(f, out) for f in SOURCES + LICENSES] + [(f, out_r) for f in SOURCES_RESNET + LICENSES] +
                  [(f, INC) for f in sorted(headers) + LICENSES])
        for f, d in placed:
            os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
            shutil.copy2(os.path.join(tree, f), os.path.join(d, f))
    where = {out: "sw/applications/tflm_mlperf/tflm/", out_r: "sw/applications/tflm_resnet/tflm/", INC: "third_party/tflm/"}
    with open(os.path.join(INC, "VENDORED.md"), "w") as fh:
        fh.write("# TensorFlow Lite Micro, the part the CPU-only baselines link\n\n"
                 f"From https://github.com/tensorflow/tflite-micro at commit `{COMMIT}`\n"
                 "(2026-11-12 23:18:28 UTC; the tflite-micro Python package 0.dev20261112231828 that made\n"
                 "`mlperf/reference_kws01.json` is built from it). Made by `mlperf/tflm_vendor.py`: TFLM's own\n"
                 f"generator (`{gen[1]} {gen[2]} <dir>`), then the {len(SOURCES)} sources the KWS firmware links\n"
                 f"(in `sw/applications/tflm_mlperf/tflm/`), the {len(SOURCES_RESNET)} sources the ResNet8 firmware links (in\n"
                 f"`sw/applications/tflm_resnet/tflm/`: the same without the depthwise kernel, with the ADD kernel) and the\n"
                 f"{len(headers)} headers they include (here). The headers are\n"
                 "kept outside `sw/`: X-HEEP's build puts every directory under an app that holds a header on the include\n"
                 "path, and FlatBuffers' `flatbuffers/string.h` would then hide `<string.h>`. Unchanged; Apache License 2.0\n"
                 "(`LICENSE`, `third_party/flatbuffers/LICENSE`, in all three places).\n\n"
                 "| sha256 | file |\n|---|---|\n")
        for f, d in placed:
            fh.write(f"| `{sha(os.path.join(d, f))[:16]}` | `{where[d]}{f}` |\n")
    n = model_cpp(os.path.join(APP, "kws_model_data.cpp"), model, "kws", "g_kws_model_data", "kws_model_data.h",
                  "MLPerf Tiny's KWS model (mlcommons/tiny benchmark/training/keyword_spotting/trained_models/\n"
                  "// kws_ref_model.tflite)")
    n_r = model_cpp(os.path.join(APP_RESNET, "resnet_model_data.cpp"), model_r, "resnet", "g_resnet_model_data",
                    "resnet_model_data.h", "MLPerf Tiny's ResNet8 (mlcommons/tiny benchmark/training/image_classification/\n"
                    "// trained_models/pretrainedResnet_quant.tflite)")
    print(f"vendored {len(SOURCES)} + {len(SOURCES_RESNET)} sources into {os.path.relpath(out)}, {os.path.relpath(out_r)} and "
          f"{len(headers)} headers into {os.path.relpath(INC)}; models {n} and {n_r} bytes")


if __name__ == "__main__":
    main()
