// RUN: %clang --target=riscv64-unknown-linux-gnu -emit-llvm -S -o- %s | \
// RUN:     FileCheck %s

// CHECK: @x = external thread_local(localexec) global i32, align 4
extern __thread int x;

void foo(int y) {
  x += y;
}
