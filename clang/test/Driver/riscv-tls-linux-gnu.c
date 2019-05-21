// RUN: %clang -target riscv64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NOPIC %s
// RUN: %clang -target riscv64-unknown-linux-gnu -fPIC %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-PIC %s

// CHECK-NOPIC: -ftls-model=local-exec
// CHECK-PIC-NOT: -ftls-model
