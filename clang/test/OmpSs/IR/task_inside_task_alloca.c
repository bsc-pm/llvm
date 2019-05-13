// RUN: %clang_cc1 -verify -fompss-2 -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(int x) {
  int a, b, c;
  int va[x];
  #pragma oss task firstprivate(x)
  {
      int d, e, f;
      int va1[x];
      #pragma oss task
      {
          int g, h, i;
      }
  }
  #pragma oss task
  {
      int j, k, l;
  }
  int m;
}

// CHECK:        %x.addr = alloca i32, align 4
// CHECK-NEXT:   %a = alloca i32, align 4
// CHECK-NEXT:   %b = alloca i32, align 4
// CHECK-NEXT:   %c = alloca i32, align 4
// CHECK-NEXT:   %saved_stack = alloca i8*, align 8
// CHECK-NEXT:   %__vla_expr0 = alloca i64, align 8
// CHECK-NEXT:   %m = alloca i32, align 4
// CHECK-NEXT:   store i32 %x, i32* %x.addr, align 4
// CHECK-NEXT:   %0 = load i32, i32* %x.addr, align 4
// CHECK-NEXT:   %1 = zext i32 %0 to i64
// CHECK-NEXT:   %2 = call i8* @llvm.stacksave()
// CHECK-NEXT:   store i8* %2, i8** %saved_stack, align 8
// CHECK-NEXT:   %vla = alloca i32, i64 %1, align 16
// CHECK-NEXT:   store i64 %1, i64* %__vla_expr0, align 8

// CHECK:        %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32* %x.addr) ]
// CHECK-NEXT:     %d = alloca i32, align 4
// CHECK-NEXT:     %e = alloca i32, align 4
// CHECK-NEXT:     %f = alloca i32, align 4
// CHECK-NEXT:     %saved_stack1 = alloca i8*, align 8
// CHECK-NEXT:     %__vla_expr1 = alloca i64, align 8
// CHECK-NEXT:     %4 = load i32, i32* %x.addr, align 4
// CHECK-NEXT:     %5 = zext i32 %4 to i64
// CHECK-NEXT:     %6 = call i8* @llvm.stacksave()
// CHECK-NEXT:     store i8* %6, i8** %saved_stack1, align 8
// CHECK-NEXT:     %vla2 = alloca i32, i64 %5, align 16
// CHECK-NEXT:     store i64 %5, i64* %__vla_expr1, align 8

// CHECK:          %7 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT:       %g = alloca i32, align 4
// CHECK-NEXT:       %h = alloca i32, align 4
// CHECK-NEXT:       %i = alloca i32, align 4
// CHECK-NEXT:     call void @llvm.directive.region.exit(token %7)

// CHECK:        %8 = load i8*, i8** %saved_stack1, align 8
// CHECK-NEXT:   call void @llvm.stackrestore(i8* %8)

// CHECK:        call void @llvm.directive.region.exit(token %3)

// CHECK:        %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT:     %j = alloca i32, align 4
// CHECK-NEXT:     %k = alloca i32, align 4
// CHECK-NEXT:     %l = alloca i32, align 4
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %9)

// CHECK:        %10 = load i8*, i8** %saved_stack, align 8
// CHECK-NEXT:   call void @llvm.stackrestore(i8* %10)

