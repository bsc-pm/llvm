// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
int y = 0;
int &ry = y;

extern int &rz;

int main() {
  int x = 0;
  int &rx = x;
  #pragma oss task depend(in : x, rx)
  { rx++; x++; }
  #pragma oss task depend(in : y, ry)
  { ry++; y++; }
  #pragma oss task depend(in : rz)
  { rz++; }
}

// CHECK: %0 = load i32*, i32** %rx, align 8
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x), "QUAL.OSS.SHARED"(i32** %rx), "QUAL.OSS.DEP.IN"(i32* %x, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32* %0, i64 4, i64 0, i64 4) ]
// CHECK-NEXT: %2 = load i32*, i32** %rx, align 8
// CHECK-NEXT: %3 = load i32, i32* %2, align 4
// CHECK-NEXT: %inc = add nsw i32 %3, 1
// CHECK-NEXT: store i32 %inc, i32* %2, align 4
// CHECK-NEXT: %4 = load i32, i32* %x, align 4
// CHECK-NEXT: %inc1 = add nsw i32 %4, 1
// CHECK-NEXT: store i32 %inc1, i32* %x, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)

// CHECK: %5 = load i32*, i32** @ry, align 8
// CHECK-NEXT: %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* @y), "QUAL.OSS.SHARED"(i32** @ry), "QUAL.OSS.DEP.IN"(i32* @y, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32* %5, i64 4, i64 0, i64 4) ]
// CHECK-NEXT: %7 = load i32*, i32** @ry, align 8
// CHECK-NEXT: %8 = load i32, i32* %7, align 4
// CHECK-NEXT: %inc2 = add nsw i32 %8, 1
// CHECK-NEXT: store i32 %inc2, i32* %7, align 4
// CHECK-NEXT: %9 = load i32, i32* @y, align 4
// CHECK-NEXT: %inc3 = add nsw i32 %9, 1
// CHECK-NEXT: store i32 %inc3, i32* @y, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %6)

// CHECK: %10 = load i32*, i32** @rz, align 8
// CHECK-NEXT: %11 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32** @rz), "QUAL.OSS.DEP.IN"(i32* %10, i64 4, i64 0, i64 4) ]
// CHECK-NEXT: %12 = load i32*, i32** @rz, align 8
// CHECK-NEXT: %13 = load i32, i32* %12, align 4
// CHECK-NEXT: %inc4 = add nsw i32 %13, 1
// CHECK-NEXT: store i32 %inc4, i32* %12, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %11)

void foo1(int &ri) {
  #pragma oss task depend(in: ri)
  { ri++; }
}

// CHECK: %0 = load i32*, i32** %ri.addr, align 8
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32** %ri.addr), "QUAL.OSS.DEP.IN"(i32* %0, i64 4, i64 0, i64 4) ]
// CHECK-NEXT: %2 = load i32*, i32** %ri.addr, align 8
// CHECK-NEXT: %3 = load i32, i32* %2, align 4
// CHECK-NEXT: %inc = add nsw i32 %3, 1
// CHECK-NEXT: store i32 %inc, i32* %2, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)


struct S {
  static int &srx;
};
void foo2() {
  #pragma oss task depend(in: S::srx)
  { S::srx = 3; }
  static int &rx = y;
  #pragma oss task depend(in: rx)
  { rx = 3; }
}

// CHECK:  %0 = load i32*, i32** @_ZN1S3srxE, align 8
// CHECK-NEXT:  %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32** @_ZN1S3srxE), "QUAL.OSS.DEP.IN"(i32* %0, i64 4, i64 0, i64 4) ]
// CHECK-NEXT:  %2 = load i32*, i32** @_ZN1S3srxE, align 8
// CHECK-NEXT:  store i32 3, i32* %2, align 4
// CHECK-NEXT:  call void @llvm.directive.region.exit(token %1)
// CHECK-NEXT:  %3 = load i32*, i32** @_ZZ4foo2vE2rx, align 8
// CHECK-NEXT:  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32** @_ZZ4foo2vE2rx), "QUAL.OSS.DEP.IN"(i32* %3, i64 4, i64 0, i64 4) ]
// CHECK-NEXT:  %5 = load i32*, i32** @_ZZ4foo2vE2rx, align 8
// CHECK-NEXT:  store i32 3, i32* %5, align 4
// CHECK-NEXT:  call void @llvm.directive.region.exit(token %4)
