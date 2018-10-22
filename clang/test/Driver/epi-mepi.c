// RUN: not %clang --target=riscv32 -mepi -E -o- %s 2>&1 \
// RUN:     | FileCheck --check-prefix=RV32 %s
// RUN: %clang --target=riscv64 -mepi -E -o- %s 2>&1 \
// RUN:     | FileCheck --check-prefix=RV64 %s

// RV32: error: EPI extension is only valid in rv64mfad
// RV64-NOT: error

int dummy;
