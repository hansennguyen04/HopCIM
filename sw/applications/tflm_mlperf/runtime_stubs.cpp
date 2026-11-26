// Copyright 2026 HopCIM contributors. Licensed under the Apache License, Version 2.0.
//
// The one C++ runtime piece X-HEEP's runtime (sw/device/lib/runtime: operator new and delete,
// __cxa_pure_virtual, __dso_handle) does not provide for a bare-metal build without exceptions: libstdc++'s
// terminate handler, which would otherwise pull in its demangler and unwinder (about 50 KiB of code). It is
// never called while the model runs.

namespace __gnu_cxx {
void __verbose_terminate_handler() {
    for (;;) {
    }
}
}  // namespace __gnu_cxx
