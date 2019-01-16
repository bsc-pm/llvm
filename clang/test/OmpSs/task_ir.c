// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int main(void) {
  int i;
  int *pi;
  int ai[5];
  #pragma oss task shared(i, pi, ai)
  { i = *pi = ai[2]; }
}

// CHECK: call token @llvm.ompss.region.entry() [ "kind"([5 x i8] c"task\00"), "shared"(i32* %i), "shared"(i32** %pi), "shared"([5 x i32]* %ai) ]
// CHECK-NEXT: %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 2
// CHECK-NEXT: %1 = load i32, i32* %arrayidx, align 8
// CHECK-NEXT: %2 = load i32*, i32** %pi, align 8
// CHECK-NEXT: store i32 %1, i32* %2, align 4
// CHECK-NEXT: store i32 %1, i32* %i, align 4
// CHECK-NEXT: call void @llvm.ompss.region.exit(token %0)

void foo(void) {
  int i;
  int *pi;
  int ai[5];
  #pragma oss task depend(in: i, pi, ai[3])
  { i = *pi = ai[2]; }
}

// CHECK: call token @llvm.ompss.region.entry() [ "kind"([5 x i8] c"task\00"), "shared"(i32* %i), "shared"([5 x i32]* %ai), "firstprivate"(i32** %pi) ]
// CHECK-NEXT: %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 2
// CHECK-NEXT: %1 = load i32, i32* %arrayidx, align 8
// CHECK-NEXT: %2 = load i32*, i32** %pi, align 8
// CHECK-NEXT: store i32 %1, i32* %2, align 4
// CHECK-NEXT: store i32 %1, i32* %i, align 4
// CHECK-NEXT: call void @llvm.ompss.region.exit(token %0)
