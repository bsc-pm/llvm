// RUN: %clang %s --target=riscv32 -### 2>&1 \
// RUN:     | FileCheck --check-prefix=RV32 %s
// RUN: %clang %s --target=riscv32 -mepi -### 2>&1 \
// RUN:     | FileCheck --check-prefix=RV32-EPI-DEFAULT %s
// RUN: %clang %s --target=riscv32 -mepi -mno-prefer-predicate-over-epilog \
// RUN:     -### 2>&1 | FileCheck --check-prefix=RV32-EPI-DISABLE %s
// RUN: %clang %s --target=riscv64 -### 2>&1 \
// RUN:     | FileCheck --check-prefix=RV64 %s
// RUN: %clang %s --target=riscv64 -mepi -### 2>&1 \
// RUN:     | FileCheck --check-prefix=RV64-EPI-DEFAULT %s
// RUN: %clang %s --target=riscv64 -mepi -mno-prefer-predicate-over-epilog \
// RUN:     -### 2>&1 | FileCheck --check-prefix=RV64-EPI-DISABLE %s

// RV32-NOT: "-prefer-predicate-over-epilog
// RV32-EPI-DEFAULT: "-mllvm" "-prefer-predicate-over-epilog=true"
// RV32-EPI-DISABLE: "-mllvm" "-prefer-predicate-over-epilog=false"
// RV64-NOT: "-prefer-predicate-over-epilog
// RV64-EPI-DEFAULT: "-mllvm" "-prefer-predicate-over-epilog=true"
// RV64-EPI-DISABLE: "-mllvm" "-prefer-predicate-over-epilog=false"
