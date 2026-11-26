// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// TensorFlow Lite Micro running MLPerf Tiny's ResNet8 (image classification), as the reference submission does
// (mlcommons/tiny benchmark/reference_submissions/image_classification/submitter_implemented.cpp): the model unchanged,
// TFLM's reference kernels (no optimised kernel library exists for this core), the reference's operators in its order.
// Differences from the reference:
//   - DepthwiseConv2D, which the reference registers but the model does not use, is left out: its code would not fit
//     the code section next to the 98,496-byte model. Registration only fills a lookup table at setup; no kernel changes;
//   - the API generation (the reference predates MicroErrorReporter's removal from the interpreter's constructor);
//   - the arena: the reference reserves 100 KiB; this layout's data section is 76 KiB, and tflm_arena_used() reports
//     what the model needs.
// C entry points for main.c.
#include <cstddef>
#include <cstdint>

#include "resnet_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#ifndef TFLM_ARENA_BYTES
#define TFLM_ARENA_BYTES (56 * 1024)
#endif

namespace {
alignas(16) uint8_t tensor_arena[TFLM_ARENA_BYTES];
tflite::MicroInterpreter *interp = nullptr;
}  // namespace

extern "C" {

// 0 when the interpreter is ready; a negative step number otherwise
int tflm_setup(void) {
    static tflite::MicroMutableOpResolver<6> resolver;   // the reference's operators in its order, without DepthwiseConv2D
    if (resolver.AddAdd() != kTfLiteOk || resolver.AddFullyConnected() != kTfLiteOk || resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk || resolver.AddSoftmax() != kTfLiteOk || resolver.AddAveragePool2D() != kTfLiteOk)
        return -1;
    const tflite::Model *model = tflite::GetModel(g_resnet_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) return -2;
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TFLM_ARENA_BYTES);
    if (interpreter.AllocateTensors() != kTfLiteOk) return -3;
    const TfLiteTensor *in = interpreter.input(0), *out = interpreter.output(0);
    if (in->type != kTfLiteInt8 || in->bytes != 3072 || out->type != kTfLiteInt8 || out->bytes != 10) return -4;
    interp = &interpreter;
    return 0;
}

int8_t *tflm_input(void) { return interp->input(0)->data.int8; }
const int8_t *tflm_output(void) { return interp->output(0)->data.int8; }
int tflm_invoke(void) { return interp->Invoke() == kTfLiteOk ? 0 : -1; }
size_t tflm_arena_used(void) { return interp->arena_used_bytes(); }
size_t tflm_arena_size(void) { return TFLM_ARENA_BYTES; }

}  // extern "C"
