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

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1), "QUAL.OSS.DEP.OUT"(%struct.S* %this1, [2 x i8] c"x\00", %struct._depend_unpack_t (%struct.S*)* @compute_dep, %struct.S* %this1) ]
// CHECK: %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1), "QUAL.OSS.DEP.IN"(%struct.S* %this1, [8 x i8] c"this->x\00", %struct._depend_unpack_t.0 (%struct.S*)* @compute_dep.1, %struct.S* %this1) ]
// CHECK: %2 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1) ]
// CHECK-NEXT: %x = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
// CHECK-NEXT: store i32 3, i32* %x, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %2)
// CHECK-NEXT: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(%struct.S* %this1) ]
// CHECK-NEXT: %x2 = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
// CHECK-NEXT: store i32 3, i32* %x2, align 4
// CHECK-NEXT: call void @llvm.directive.region.exit(token %3)


// CHECK: define internal %struct._depend_unpack_t @compute_dep(%struct.S* %this)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK-NEXT:   %this.addr = alloca %struct.S*, align 8
// CHECK-NEXT:   store %struct.S* %this, %struct.S** %this.addr, align 8
// CHECK-NEXT:   %this1 = load %struct.S*, %struct.S** %this.addr, align 8
// CHECK-NEXT:   %x = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
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

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1(%struct.S* %this)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK-NEXT:   %this.addr = alloca %struct.S*, align 8
// CHECK-NEXT:   store %struct.S* %this, %struct.S** %this.addr, align 8
// CHECK-NEXT:   %this1 = load %struct.S*, %struct.S** %this.addr, align 8
// CHECK-NEXT:   %x = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
// CHECK-NEXT:   %0 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %x, i32** %0, align 8
// CHECK-NEXT:   %1 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 4, i64* %1, align 8
// CHECK-NEXT:   %2 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 0, i64* %2, align 8
// CHECK-NEXT:   %3 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 4, i64* %3, align 8
// CHECK-NEXT:   %4 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %4
// CHECK-NEXT: }
