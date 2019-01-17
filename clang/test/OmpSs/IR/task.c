// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int main(void) {
  int i;
  int *pi;
  int ai[5];
  #pragma oss task shared(i, pi, ai)
  { i = *pi = ai[2]; }
}

// CHECK: call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %i), "QUAL.OSS.SHARED"(i32** %pi), "QUAL.OSS.SHARED"([5 x i32]* %ai) ]
// CHECK-NEXT: %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 2
// CHECK-NEXT: %1 = load i32, i32* %arrayidx, align 8
// CHECK-NEXT: %2 = load i32*, i32** %pi, align 8
// CHECK-NEXT: store i32 %1, i32* %2, align 4
// CHECK-NEXT: store i32 %1, i32* %i, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)

void foo(void) {
  int i;
  int *pi;
  int ai[5];
  #pragma oss task depend(in: i, pi, ai[3])
  { i = *pi = ai[2]; }
}

// CHECK: call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %i), "QUAL.OSS.SHARED"([5 x i32]* %ai), "QUAL.OSS.FIRSTPRIVATE"(i32** %pi) ]
// CHECK-NEXT: %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 2
// CHECK-NEXT: %1 = load i32, i32* %arrayidx, align 8
// CHECK-NEXT: %2 = load i32*, i32** %pi, align 8
// CHECK-NEXT: store i32 %1, i32* %2, align 4
// CHECK-NEXT: store i32 %1, i32* %i, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)

struct Foo1_struct {
    int x;
} foo1_s;

int foo1_array[5];
int foo1_var;
int *foo1_ptr;

void foo1(void) {
  #pragma oss task depend(in: foo1_var, *foo1_ptr, foo1_array[3], foo1_s.x)
  { foo1_var = *foo1_ptr = foo1_array[3] = foo1_s.x; }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* @foo1_var), "QUAL.OSS.SHARED"([5 x i32]* @foo1_array), "QUAL.OSS.SHARED"(%struct.Foo1_struct* @foo1_s), "QUAL.OSS.FIRSTPRIVATE"(i32** @foo1_ptr) ]
// CHECK-NEXT: %1 = load i32, i32* getelementptr inbounds (%struct.Foo1_struct, %struct.Foo1_struct* @foo1_s, i32 0, i32 0), align 4
// CHECK-NEXT: store i32 %1, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo1_array, i64 0, i64 3), align 4
// CHECK-NEXT: %2 = load i32*, i32** @foo1_ptr, align 8
// CHECK-NEXT: store i32 %1, i32* %2, align 4
// CHECK-NEXT: store i32 %1, i32* @foo1_var, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)
