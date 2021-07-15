// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(void) {
  int i;
  int *pi;
  int ai[5];
  #pragma oss task depend(in: i, pi, ai[3])
  { i = *pi = ai[2]; }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %i), "QUAL.OSS.SHARED"(i32** %pi), "QUAL.OSS.SHARED"([5 x i32]* %ai), "QUAL.OSS.DEP.IN"(i32* %i, [2 x i8] c"i\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %i), "QUAL.OSS.DEP.IN"(i32** %pi, [3 x i8] c"pi\00", %struct._depend_unpack_t.0 (i32**)* @compute_dep.1, i32** %pi), "QUAL.OSS.DEP.IN"([5 x i32]* %ai, [6 x i8] c"ai[3]\00", %struct._depend_unpack_t.1 ([5 x i32]*)* @compute_dep.2, [5 x i32]* %ai) ]
// CHECK-NEXT: %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 2
// CHECK-NEXT: %1 = load i32, i32* %arrayidx, align
// CHECK-NEXT: %2 = load i32*, i32** %pi, align 8
// CHECK-NEXT: store i32 %1, i32* %2, align 4
// CHECK-NEXT: store i32 %1, i32* %i, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32* %i)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %i, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1(i32** %pi)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32** %pi, i32*** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 8, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 8, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.2([5 x i32]* %ai)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.1, align 8
// CHECK:   %arraydecay = getelementptr inbounds [5 x i32], [5 x i32]* %ai, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arraydecay, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 20, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 12, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 16, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %4
// CHECK-NEXT: }

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

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* @foo1_var), "QUAL.OSS.SHARED"([5 x i32]* @foo1_array), "QUAL.OSS.SHARED"(%struct.Foo1_struct* @foo1_s), "QUAL.OSS.FIRSTPRIVATE"(i32** @foo1_ptr), "QUAL.OSS.DEP.IN"(i32* @foo1_var, [9 x i8] c"foo1_var\00", %struct._depend_unpack_t.2 (i32*)* @compute_dep.3, i32* @foo1_var), "QUAL.OSS.DEP.IN"(i32** @foo1_ptr, [10 x i8] c"*foo1_ptr\00", %struct._depend_unpack_t.3 (i32**)* @compute_dep.4, i32** @foo1_ptr), "QUAL.OSS.DEP.IN"([5 x i32]* @foo1_array, [14 x i8] c"foo1_array[3]\00", %struct._depend_unpack_t.4 ([5 x i32]*)* @compute_dep.5, [5 x i32]* @foo1_array), "QUAL.OSS.DEP.IN"([5 x i32]* @foo1_array, [15 x i8] c"foo1_array[-2]\00", %struct._depend_unpack_t.5 ([5 x i32]*)* @compute_dep.6, [5 x i32]* @foo1_array), "QUAL.OSS.DEP.IN"(%struct.Foo1_struct* @foo1_s, [9 x i8] c"foo1_s.x\00", %struct._depend_unpack_t.6 (%struct.Foo1_struct*)* @compute_dep.7, %struct.Foo1_struct* @foo1_s) ]
// CHECK-NEXT: %1 = load i32, i32* getelementptr inbounds (%struct.Foo1_struct, %struct.Foo1_struct* @foo1_s, i32 0, i32 0), align 4
// CHECK-NEXT: store i32 %1, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @foo1_array, i64 0, i64 3), align 4
// CHECK-NEXT: %2 = load i32*, i32** @foo1_ptr, align 8
// CHECK-NEXT: store i32 %1, i32* %2, align 4
// CHECK-NEXT: store i32 %1, i32* @foo1_var, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)

// CHECK: define internal %struct._depend_unpack_t.2 @compute_dep.3(i32* %foo1_var)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.2, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %foo1_var, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.2 %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.3 @compute_dep.4(i32** %foo1_ptr)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.3, align 8
// CHECK:   %0 = load i32*, i32** %foo1_ptr, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %0, i32** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.3 %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.4 @compute_dep.5([5 x i32]* %foo1_array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.4, align 8
// CHECK:   %arraydecay = getelementptr inbounds [5 x i32], [5 x i32]* %foo1_array, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arraydecay, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 20, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 12, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 16, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.4 %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.5 @compute_dep.6([5 x i32]* %foo1_array)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.5, align 8
// CHECK:   %arraydecay = getelementptr inbounds [5 x i32], [5 x i32]* %foo1_array, i64 0, i64 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arraydecay, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 20, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 -8, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 -4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.5 %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.6 @compute_dep.7(%struct.Foo1_struct* %foo1_s)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.6, align 8
// CHECK:   %x = getelementptr inbounds %struct.Foo1_struct, %struct.Foo1_struct* %foo1_s, i32 0, i32 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %x, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.6 %4
// CHECK-NEXT: }

void foo2(int *iptr, char *cptr) {
  #pragma oss task depend(in: iptr[3], iptr[-3], cptr[3], cptr[-3])
  { iptr[3] = cptr[3]; }
  #pragma oss task depend(in: *iptr, *cptr)
  { *iptr = *cptr; }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %iptr.addr), "QUAL.OSS.FIRSTPRIVATE"(i8** %cptr.addr), "QUAL.OSS.DEP.IN"(i32** %iptr.addr, [8 x i8] c"iptr[3]\00", %struct._depend_unpack_t.7 (i32**)* @compute_dep.8, i32** %iptr.addr), "QUAL.OSS.DEP.IN"(i32** %iptr.addr, [9 x i8] c"iptr[-3]\00", %struct._depend_unpack_t.8 (i32**)* @compute_dep.9, i32** %iptr.addr), "QUAL.OSS.DEP.IN"(i8** %cptr.addr, [8 x i8] c"cptr[3]\00", %struct._depend_unpack_t.9 (i8**)* @compute_dep.10, i8** %cptr.addr), "QUAL.OSS.DEP.IN"(i8** %cptr.addr, [9 x i8] c"cptr[-3]\00", %struct._depend_unpack_t.10 (i8**)* @compute_dep.11, i8** %cptr.addr) ]
// CHECK-NEXT: %1 = load i8*, i8** %cptr.addr, align 8
// CHECK-NEXT: %arrayidx = getelementptr inbounds i8, i8* %1, i64 3
// CHECK-NEXT: %2 = load i8, i8* %arrayidx, align 1
// CHECK-NEXT: %conv = sext i8 %2 to i32
// CHECK-NEXT: %3 = load i32*, i32** %iptr.addr, align 8
// CHECK-NEXT: %arrayidx1 = getelementptr inbounds i32, i32* %3, i64 3
// CHECK-NEXT: store i32 %conv, i32* %arrayidx1, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)

// CHECK: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(i32** %iptr.addr), "QUAL.OSS.FIRSTPRIVATE"(i8** %cptr.addr), "QUAL.OSS.DEP.IN"(i32** %iptr.addr, [6 x i8] c"*iptr\00", %struct._depend_unpack_t.11 (i32**)* @compute_dep.12, i32** %iptr.addr), "QUAL.OSS.DEP.IN"(i8** %cptr.addr, [6 x i8] c"*cptr\00", %struct._depend_unpack_t.12 (i8**)* @compute_dep.13, i8** %cptr.addr) ]
// CHECK-NEXT: %5 = load i8*, i8** %cptr.addr, align 8
// CHECK-NEXT: %6 = load i8, i8* %5, align 1
// CHECK-NEXT: %conv2 = sext i8 %6 to i32
// CHECK-NEXT: %7 = load i32*, i32** %iptr.addr, align 8
// CHECK-NEXT: store i32 %conv2, i32* %7, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %4)

// CHECK: define internal %struct._depend_unpack_t.7 @compute_dep.8(i32** %iptr)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.7, align 8
// CHECK:   %0 = load i32*, i32** %iptr, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %0, i32** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 12, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 16, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.7, %struct._depend_unpack_t.7* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.7 %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.8 @compute_dep.9(i32** %iptr)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.8, align 8
// CHECK:   %0 = load i32*, i32** %iptr, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %0, i32** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 -12, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 -8, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.8, %struct._depend_unpack_t.8* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.8 %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.9 @compute_dep.10(i8** %cptr)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.9, align 8
// CHECK:   %0 = load i8*, i8** %cptr, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 0
// CHECK-NEXT:   store i8* %0, i8** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 1, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 3, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.9, %struct._depend_unpack_t.9* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.9 %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.10 @compute_dep.11(i8** %cptr)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.10, align 8
// CHECK:   %0 = load i8*, i8** %cptr, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 0
// CHECK-NEXT:   store i8* %0, i8** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 1, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 -3, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 -2, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.10, %struct._depend_unpack_t.10* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.10 %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.11 @compute_dep.12(i32** %iptr)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.11, align 8
// CHECK:   %0 = load i32*, i32** %iptr, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %0, i32** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.11, %struct._depend_unpack_t.11* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.11 %5
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.12 @compute_dep.13(i8** %cptr)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.12, align 8
// CHECK:   %0 = load i8*, i8** %cptr, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 0
// CHECK-NEXT:   store i8* %0, i8** %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 1, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %3, align 8
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 1, i64* %4, align 8
// CHECK-NEXT:   %5 = load %struct._depend_unpack_t.12, %struct._depend_unpack_t.12* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.12 %5
// CHECK-NEXT: }

struct Foo3_struct {
    int x;
};

void foo3() {
  struct Foo3_struct foo3_struct;
  int a;
  #pragma oss task depend(in: foo3_struct.x, a)
  { foo3_struct.x = a; }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.Foo3_struct* %foo3_struct), "QUAL.OSS.SHARED"(i32* %a), "QUAL.OSS.DEP.IN"(%struct.Foo3_struct* %foo3_struct, [14 x i8] c"foo3_struct.x\00", %struct._depend_unpack_t.13 (%struct.Foo3_struct*)* @compute_dep.14, %struct.Foo3_struct* %foo3_struct), "QUAL.OSS.DEP.IN"(i32* %a, [2 x i8] c"a\00", %struct._depend_unpack_t.14 (i32*)* @compute_dep.15, i32* %a) ]
// CHECK-NEXT: %1 = load i32, i32* %a, align 4
// CHECK-NEXT: %x = getelementptr inbounds %struct.Foo3_struct, %struct.Foo3_struct* %foo3_struct, i32 0, i32 0
// CHECK-NEXT: store i32 %1, i32* %x, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %0)

// CHECK: define internal %struct._depend_unpack_t.13 @compute_dep.14(%struct.Foo3_struct* %foo3_struct)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.13, align 8
// CHECK:   %x = getelementptr inbounds %struct.Foo3_struct, %struct.Foo3_struct* %foo3_struct, i32 0, i32 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %x, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.13, %struct._depend_unpack_t.13* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.13 %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.14 @compute_dep.15(i32* %a)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.14, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %a, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.14, %struct._depend_unpack_t.14* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.14 %4
// CHECK-NEXT: }
