// RUN: not %clang --target=riscv32 -mepi -E -o- %s 2>&1 \
// RUN:     | FileCheck --check-prefix=ERROR %s
// RUN: not %clang --target=riscv64 -mepi -E -o- %s 2>&1 \
// RUN:     | FileCheck --check-prefix=ERROR %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -mepi -E -o- %s 2>&1 \
// RUN:     | FileCheck --check-prefix=GOOD %s

// ERROR: error: EPI extension is only valid in rv64mfad
// GOOD-NOT: error

int dummy;
