// RUN: %clang_cc1 -verify -fompss-2 -fexceptions -fcxx-exceptions -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#pragma oss task
void foo() {
    throw 43;
}

int main() {
    foo();
    foo();
}

// CHECK:   %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.DECL.SOURCE"([8 x i8] c"foo:4:9\00") ]
// CHECK-NEXT:   invoke void @_Z3foov()
// CHECK-NEXT:           to label %invoke.cont unwind label %terminate.lpad
// CHECK: invoke.cont:                                      ; preds = %entry
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %0)
// CHECK-NEXT:   %1 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.DECL.SOURCE"([8 x i8] c"foo:4:9\00") ]
// CHECK-NEXT:   invoke void @_Z3foov()
// CHECK-NEXT:           to label %invoke.cont1 unwind label %terminate.lpad2
// CHECK: invoke.cont1:                                     ; preds = %invoke.cont
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %1)
// CHECK-NEXT:   ret i32 0
// CHECK: terminate.lpad:                                   ; preds = %entry
// CHECK-NEXT:   %2 = landingpad { i8*, i32 }
// CHECK-NEXT:           catch i8* null
// CHECK-NEXT:   %3 = extractvalue { i8*, i32 } %2, 0
// CHECK-NEXT:   call void @__clang_call_terminate(i8* %3) #4
// CHECK-NEXT:   unreachable
// CHECK: terminate.lpad2:                                  ; preds = %invoke.cont
// CHECK-NEXT:   %4 = landingpad { i8*, i32 }
// CHECK-NEXT:           catch i8* null
// CHECK-NEXT:   %5 = extractvalue { i8*, i32 } %4, 0
// CHECK-NEXT:   call void @__clang_call_terminate(i8* %5) #4
// CHECK-NEXT:   unreachable

