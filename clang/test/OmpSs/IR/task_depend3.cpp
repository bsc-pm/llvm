// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
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
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x), "QUAL.OSS.SHARED"(i32* %0), "QUAL.OSS.DEP.IN"(i32* %x, [2 x i8] c"x\00", %struct._depend_unpack_t (i32*)* @compute_dep, i32* %x), "QUAL.OSS.DEP.IN"(i32* %0, [3 x i8] c"rx\00", %struct._depend_unpack_t.0 (i32*)* @compute_dep.1, i32* %0) ]
// CHECK-NEXT: %2 = load i32, i32* %0, align 4
// CHECK-NEXT: %inc = add nsw i32 %2, 1
// CHECK-NEXT: store i32 %inc, i32* %0, align 4
// CHECK-NEXT: %3 = load i32, i32* %x, align 4
// CHECK-NEXT: %inc1 = add nsw i32 %3, 1
// CHECK-NEXT: store i32 %inc1, i32* %x, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)

// CHECK-NEXT: %4 = load i32*, i32** @ry, align 8
// CHECK-NEXT: %5 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* @y), "QUAL.OSS.SHARED"(i32* %4), "QUAL.OSS.DEP.IN"(i32* @y, [2 x i8] c"y\00", %struct._depend_unpack_t.1 (i32*)* @compute_dep.2, i32* @y), "QUAL.OSS.DEP.IN"(i32* %4, [3 x i8] c"ry\00", %struct._depend_unpack_t.2 (i32*)* @compute_dep.3, i32* %4) ]
// CHECK-NEXT: %6 = load i32, i32* %4, align 4
// CHECK-NEXT: %inc2 = add nsw i32 %6, 1
// CHECK-NEXT: store i32 %inc2, i32* %4, align 4
// CHECK-NEXT: %7 = load i32, i32* @y, align 4
// CHECK-NEXT: %inc3 = add nsw i32 %7, 1
// CHECK-NEXT: store i32 %inc3, i32* @y, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %5)

// CHECK-NEXT: %8 = load i32*, i32** @rz, align 8
// CHECK-NEXT: %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %8), "QUAL.OSS.DEP.IN"(i32* %8, [3 x i8] c"rz\00", %struct._depend_unpack_t.3 (i32*)* @compute_dep.4, i32* %8) ]
// CHECK-NEXT: %10 = load i32, i32* %8, align 4
// CHECK-NEXT: %inc4 = add nsw i32 %10, 1
// CHECK-NEXT: store i32 %inc4, i32* %8, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %9)

// CHECK: define internal %struct._depend_unpack_t @compute_dep(i32* %x)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %x, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1(i32* %rx)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %rx, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.1 @compute_dep.2(i32* %y)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.1, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %y, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.1, %struct._depend_unpack_t.1* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.1 %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.2 @compute_dep.3(i32* %ry)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.2, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %ry, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.2, %struct._depend_unpack_t.2* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.2 %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.3 @compute_dep.4(i32* %rz)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.3, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %rz, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.3, %struct._depend_unpack_t.3* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.3 %4
// CHECK-NEXT: }

void foo1(int &ri) {
  #pragma oss task depend(in: ri)
  { ri++; }
}

// CHECK: %0 = load i32*, i32** %ri.addr, align 8
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %0), "QUAL.OSS.DEP.IN"(i32* %0, [3 x i8] c"ri\00", %struct._depend_unpack_t.4 (i32*)* @compute_dep.5, i32* %0) ]
// CHECK-NEXT: %2 = load i32, i32* %0, align 4
// CHECK-NEXT: %inc = add nsw i32 %2, 1
// CHECK-NEXT: store i32 %inc, i32* %0, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)

// CHECK: define internal %struct._depend_unpack_t.4 @compute_dep.5(i32* %ri)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.4, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %ri, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.4, %struct._depend_unpack_t.4* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.4 %4
// CHECK-NEXT: }

struct S {
  static int &srx;
};
void foo2() {
  #pragma oss task depend(in: S::srx)
  { S::srx = 3; }
  static int &rx = y;
  #pragma oss task depend(in: rx)
  { rx = 3; }
  #pragma oss task
  {
      int x;
      extern int &rex;
      static int &rsx = x;
      rex = rsx;
  }
}

// CHECK: %0 = load i32*, i32** @_ZN1S3srxE, align 8
// CHECK-NEXT: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %0), "QUAL.OSS.DEP.IN"(i32* %0, [7 x i8] c"S::srx\00", %struct._depend_unpack_t.5 (i32*)* @compute_dep.6, i32* %0) ]
// CHECK-NEXT: store i32 3, i32* %0, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %1)

// CHECK-NEXT: %2 = load i32*, i32** @_ZZ4foo2vE2rx, align 8
// CHECK-NEXT: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %2), "QUAL.OSS.DEP.IN"(i32* %2, [3 x i8] c"rx\00", %struct._depend_unpack_t.6 (i32*)* @compute_dep.7, i32* %2) ]
// CHECK-NEXT: store i32 3, i32* %2, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %3)

// CHECK-NEXT: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT: %x = alloca i32, align 4
// CHECK-NEXT: %5 = load atomic i8, i8* bitcast (i64* @_ZGVZ4foo2vE3rsx to i8*) acquire, align 8
// CHECK-NEXT: %guard.uninitialized = icmp eq i8 %5, 0
// CHECK-NEXT: br i1 %guard.uninitialized, label %init.check, label %init.end
// CHECK: init.check:                                       ; preds = %entry
// CHECK-NEXT:   %6 = call i32 @__cxa_guard_acquire(i64* @_ZGVZ4foo2vE3rsx) #1
// CHECK-NEXT:   %tobool = icmp ne i32 %6, 0
// CHECK-NEXT:   br i1 %tobool, label %init, label %init.end
// CHECK: init:                                             ; preds = %init.check
// CHECK-NEXT:   store i32* %x, i32** @_ZZ4foo2vE3rsx, align 8
// CHECK-NEXT:   call void @__cxa_guard_release(i64* @_ZGVZ4foo2vE3rsx) #1
// CHECK-NEXT:   br label %init.end
// CHECK: init.end:                                         ; preds = %init, %init.check, %entry
// CHECK-NEXT:   %7 = load i32*, i32** @_ZZ4foo2vE3rsx, align 8
// CHECK-NEXT:   %8 = load i32, i32* %7, align 4
// CHECK-NEXT:   %9 = load i32*, i32** @rex, align 8
// CHECK-NEXT:   store i32 %8, i32* %9, align 4
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %4)

// CHECK: define internal %struct._depend_unpack_t.5 @compute_dep.6(i32* %srx)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.5, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %srx, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.5, %struct._depend_unpack_t.5* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.5 %4
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.6 @compute_dep.7(i32* %rx)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.6, align 8
// CHECK:   %0 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %rx, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.6, %struct._depend_unpack_t.6* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.6 %4
// CHECK-NEXT: }
