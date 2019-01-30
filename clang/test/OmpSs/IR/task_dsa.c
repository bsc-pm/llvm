// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

struct A {
    int x;
};

int main(void) {
  int i;
  int *pi;
  int ai[5];
  struct A sa;
  #pragma oss task shared(i, pi, ai, sa)
  { i = *pi = ai[2] = sa.x; }
  #pragma oss task private(i, pi, ai, sa)
  { i = *pi = ai[2] = sa.x; }
  #pragma oss task firstprivate(i, pi, ai, sa)
  { i = *pi = ai[2] = sa.x; }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %i), "QUAL.OSS.SHARED"(i32** %pi), "QUAL.OSS.SHARED"([5 x i32]* %ai), "QUAL.OSS.SHARED"(%struct.A* %sa) ]
// CHECK-NEXT: %x = getelementptr inbounds %struct.A, %struct.A* %sa, i32 0, i32 0
// CHECK-NEXT: %1 = load i32, i32* %x, align 4
// CHECK-NEXT: %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 2
// CHECK-NEXT: store i32 %1, i32* %arrayidx, align 8
// CHECK-NEXT: %2 = load i32*, i32** %pi, align 8
// CHECK-NEXT: store i32 %1, i32* %2, align 4
// CHECK-NEXT: store i32 %1, i32* %i, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)

// CHECK-NEXT: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.PRIVATE"(i32** %pi), "QUAL.OSS.PRIVATE"([5 x i32]* %ai), "QUAL.OSS.PRIVATE"(%struct.A* %sa) ]
// CHECK-NEXT: %x1 = getelementptr inbounds %struct.A, %struct.A* %sa, i32 0, i32 0
// CHECK-NEXT: %4 = load i32, i32* %x1, align 4
// CHECK-NEXT: %arrayidx2 = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 2
// CHECK-NEXT: store i32 %4, i32* %arrayidx2, align 8
// CHECK-NEXT: %5 = load i32*, i32** %pi, align 8
// CHECK-NEXT: store i32 %4, i32* %5, align 4
// CHECK-NEXT: store i32 %4, i32* %i, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %3)

// CHECK-NEXT: %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %i), "QUAL.OSS.FIRSTPRIVATE"(i32** %pi), "QUAL.OSS.FIRSTPRIVATE"([5 x i32]* %ai), "QUAL.OSS.FIRSTPRIVATE"(%struct.A* %sa) ]
// CHECK-NEXT: %x3 = getelementptr inbounds %struct.A, %struct.A* %sa, i32 0, i32 0
// CHECK-NEXT: %7 = load i32, i32* %x3, align 4
// CHECK-NEXT: %arrayidx4 = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 2
// CHECK-NEXT: store i32 %7, i32* %arrayidx4, align 8
// CHECK-NEXT: %8 = load i32*, i32** %pi, align 8
// CHECK-NEXT: store i32 %7, i32* %8, align 4
// CHECK-NEXT: store i32 %7, i32* %i, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %6)
