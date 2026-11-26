// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// TensorFlow Lite Micro running MLPerf Tiny's KWS model, as the reference submission does
// (mlcommons/tiny benchmark/reference_submissions/keyword_spotting/submitter_implemented.cpp and
// benchmark/util/tf_micro_model_runner.h): the same six operators in the same order, the model unchanged,
// TFLM's reference kernels (no optimised kernel library exists for this core). C entry points for main.c.
// The differences from the reference are the API generation (the reference predates MicroErrorReporter's
// removal from the interpreter's constructor) and the arena size (the reference reserves 200 KiB; this
// board has 128 KiB of data RAM, and tflm_arena_used() reports what the model needs).
#include <cstddef>
#include <cstdint>

#include "kws_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#ifndef TFLM_ARENA_BYTES
#define TFLM_ARENA_BYTES (64 * 1024)
#endif

namespace {
alignas(16) uint8_t tensor_arena[TFLM_ARENA_BYTES];
tflite::MicroInterpreter *interp = nullptr;
}  // namespace

extern "C" {

// 0 when the interpreter is ready; a negative step number otherwise
int tflm_setup(void) {
    static tflite::MicroMutableOpResolver<6> resolver;   // the reference's operators, in its order
    if (resolver.AddFullyConnected() != kTfLiteOk || resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk || resolver.AddReshape() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk || resolver.AddAveragePool2D() != kTfLiteOk)
        return -1;
    const tflite::Model *model = tflite::GetModel(g_kws_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) return -2;
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TFLM_ARENA_BYTES);
    if (interpreter.AllocateTensors() != kTfLiteOk) return -3;
    const TfLiteTensor *in = interpreter.input(0), *out = interpreter.output(0);
    if (in->type != kTfLiteInt8 || in->bytes != 490 || out->type != kTfLiteInt8 || out->bytes != 12) return -4;
    interp = &interpreter;
    return 0;
}

int8_t *tflm_input(void) { return interp->input(0)->data.int8; }
const int8_t *tflm_output(void) { return interp->output(0)->data.int8; }
float tflm_output_scale(void) { return interp->output(0)->params.scale; }
int tflm_output_zero_point(void) { return interp->output(0)->params.zero_point; }
int tflm_invoke(void) { return interp->Invoke() == kTfLiteOk ? 0 : -1; }
size_t tflm_arena_used(void) { return interp->arena_used_bytes(); }
size_t tflm_arena_size(void) { return TFLM_ARENA_BYTES; }

}  // extern "C"
