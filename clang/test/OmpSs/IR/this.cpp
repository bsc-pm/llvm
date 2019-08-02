// RUN: %clang_cc1 -x c++ -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

struct S {
  int x = 43;
  int y;
  void f();
};

void S::f() {
  #pragma oss task depend(out : x)
  {}
  #pragma oss task depend(in : this->x)
  {}
  #pragma oss task
  { x = 3; }
  #pragma oss task
  { this->x = 3; }
}

// CHECK: %this1 = load %struct.S*, %struct.S** %this.addr, align 8
// CHECK-NEXT: %x = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
// CHECK-NEXT: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1), "QUAL.OSS.DEP.OUT"(i32* %x, i64 4, i64 0, i64 4) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)
// CHECK-NEXT: %x2 = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1), "QUAL.OSS.DEP.IN"(i32* %x2, i64 4, i64 0, i64 4) ]
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)
// CHECK-NEXT: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1) ]
// CHECK-NEXT: %x3 = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
// CHECK-NEXT: store i32 3, i32* %x3, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %2)
// CHECK-NEXT: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1) ]
// CHECK-NEXT: %x4 = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
// CHECK-NEXT: store i32 3, i32* %x4, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %3)

