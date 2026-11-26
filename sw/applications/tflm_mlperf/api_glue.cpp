// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
// C entry points to MLCommons' unchanged internally_implemented.cpp (C++ linkage), for main.c.
#include "api/internally_implemented.h"

extern "C" {
void hop_ee_benchmark_initialize(void) { ee_benchmark_initialize(); }
void hop_ee_serial_callback(char c) { ee_serial_callback(c); }
void hop_ee_command(char *command) { ee_serial_command_parser_callback(command); }
size_t hop_ee_get_buffer(uint8_t *buffer, size_t max_len) { return ee_get_buffer(buffer, max_len); }
}
