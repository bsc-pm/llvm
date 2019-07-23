// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(void) {
  int i;
  int *pi;
  int ai[5];
  #pragma oss task depend(in: i, pi, ai[3])
  { i = *pi = ai[2]; }
}

// CHECK: %arraydecay = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 0
// CHECK-NEXT: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %i), "QUAL.OSS.SHARED"([5 x i32]* %ai), "QUAL.OSS.FIRSTPRIVATE"(i32** %pi), "QUAL.OSS.DEP.IN"(i32* %i, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32** %pi, i64 8, i64 0, i64 8), "QUAL.OSS.DEP.IN"(i32* %arraydecay, i64 20, i64 12, i64 16) ]
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
  #pragma oss task depend(in: foo1_var, *foo1_ptr, foo1_array[3], foo1_array[-2], foo1_s.x)
  { foo1_var = *foo1_ptr = foo1_array[3] = foo1_s.x; }
}

// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* @foo1_var), "QUAL.OSS.SHARED"([5 x i32]* @foo1_array), "QUAL.OSS.SHARED"(%struct.Foo1_struct* @foo1_s), "QUAL.OSS.FIRSTPRIVATE"(i32** @foo1_ptr), "QUAL.OSS.DEP.IN"(i32* @foo1_var, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32* %0, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo1_array, i64 0, i64 0), i64 20, i64 12, i64 16), "QUAL.OSS.DEP.IN"(i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo1_array, i64 0, i64 0), i64 20, i64 -8, i64 -4), "QUAL.OSS.DEP.IN"(i32* getelementptr inbounds (%struct.Foo1_struct, %struct.Foo1_struct* @foo1_s, i32 0, i32 0), i64 4, i64 0, i64 4) ]
// CHECK-NEXT: %2 = load i32, i32* getelementptr inbounds (%struct.Foo1_struct, %struct.Foo1_struct* @foo1_s, i32 0, i32 0), align 4
// CHECK-NEXT: store i32 %2, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo1_array, i64 0, i64 3), align 4
// CHECK-NEXT: %3 = load i32*, i32** @foo1_ptr, align 8
// CHECK-NEXT: store i32 %2, i32* %3, align 4
// CHECK-NEXT: store i32 %2, i32* @foo1_var, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)

void foo2(int *iptr, char *cptr) {
  #pragma oss task depend(in: iptr[3], iptr[-3], cptr[3], cptr[-3])
  { iptr[3] = cptr[3]; }
  #pragma oss task depend(in: *iptr, *cptr)
  { *iptr = *cptr; }
}

// CHECK:  %0 = load i32*, i32** %iptr.addr, align 8
// CHECK-NEXT:  %1 = load i32*, i32** %iptr.addr, align 8
// CHECK-NEXT:  %2 = load i8*, i8** %cptr.addr, align 8
// CHECK-NEXT:  %3 = load i8*, i8** %cptr.addr, align 8
// CHECK-NEXT:  %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %iptr.addr), "QUAL.OSS.FIRSTPRIVATE"(i8** %cptr.addr), "QUAL.OSS.DEP.IN"(i32* %0, i64 4, i64 12, i64 16), "QUAL.OSS.DEP.IN"(i32* %1, i64 4, i64 -12, i64 -8), "QUAL.OSS.DEP.IN"(i8* %2, i64 1, i64 3, i64 4), "QUAL.OSS.DEP.IN"(i8* %3, i64 1, i64 -3, i64 -2) ]
// CHECK-NEXT:  %5 = load i8*, i8** %cptr.addr, align 8
// CHECK-NEXT:  %arrayidx = getelementptr inbounds i8, i8* %5, i64 3
// CHECK-NEXT:  %6 = load i8, i8* %arrayidx, align 1
// CHECK-NEXT:  %conv = sext i8 %6 to i32
// CHECK-NEXT:  %7 = load i32*, i32** %iptr.addr, align 8
// CHECK-NEXT:  %arrayidx1 = getelementptr inbounds i32, i32* %7, i64 3
// CHECK-NEXT:  store i32 %conv, i32* %arrayidx1, align 4
// CHECK-NEXT:  call void @llvm.directive.region.exit(token %4)

// CHECK:  %8 = load i32*, i32** %iptr.addr, align 8
// CHECK-NEXT:  %9 = load i8*, i8** %cptr.addr, align 8
// CHECK-NEXT:  %10 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %iptr.addr), "QUAL.OSS.FIRSTPRIVATE"(i8** %cptr.addr), "QUAL.OSS.DEP.IN"(i32* %8, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i8* %9, i64 1, i64 0, i64 1) ]
// CHECK-NEXT:  %11 = load i8*, i8** %cptr.addr, align 8
// CHECK-NEXT:  %12 = load i8, i8* %11, align 1
// CHECK-NEXT:  %conv2 = sext i8 %12 to i32
// CHECK-NEXT:  %13 = load i32*, i32** %iptr.addr, align 8
// CHECK-NEXT:  store i32 %conv2, i32* %13, align 4
// CHECK-NEXT:  call void @llvm.directive.region.exit(token %10)

struct Foo3_struct {
    int x;
};

void foo3() {
  struct Foo3_struct foo3_struct;
  int a;
  #pragma oss task depend(in: foo3_struct.x, a)
  { foo3_struct.x = a; }
}

// CHECK: %x = getelementptr inbounds %struct.Foo3_struct, %struct.Foo3_struct* %foo3_struct, i32 0, i32 0
// CHECK-NEXT: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.Foo3_struct* %foo3_struct), "QUAL.OSS.SHARED"(i32* %a), "QUAL.OSS.DEP.IN"(i32* %x, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.IN"(i32* %a, i64 4, i64 0, i64 4) ]
// CHECK-NEXT: %1 = load i32, i32* %a, align 4
// CHECK-NEXT: %x1 = getelementptr inbounds %struct.Foo3_struct, %struct.Foo3_struct* %foo3_struct, i32 0, i32 0
// CHECK-NEXT: store i32 %1, i32* %x1, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)

