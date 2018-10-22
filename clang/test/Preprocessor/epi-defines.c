// RUN: %clang -target riscv64-unknown-linux-gnu -x c -E -dM %s -o - \
// RUN:      | FileCheck %s
// RUN: %clang -target riscv64-unknown-linux-gnu -mepi -x c -E -dM %s -o - \
// RUN:      | FileCheck --check-prefix=EPI %s

// CHECK-NOT: __epi
// EPI: #define __epi 1
