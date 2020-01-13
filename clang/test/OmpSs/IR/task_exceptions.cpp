// RUN: %clang_cc1 -verify -fompss-2 -fexceptions -fcxx-exceptions -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo1() {
    #pragma oss task
    {
        throw 1;
    }
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT: %exception = call i8* @__cxa_allocate_exception(i64 4)
// CHECK-NEXT: %1 = bitcast i8* %exception to i32*
// CHECK-NEXT: store i32 1, i32* %1, align 16
// CHECK-NEXT: invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
// CHECK-NEXT:         to label %invoke.cont unwind label %terminate.lpad
// CHECK: invoke.cont:                                      ; preds = %entry
// CHECK-NEXT:   br label %throw.cont
// CHECK: throw.cont:                                       ; preds = %invoke.cont
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %0)
// CHECK-NEXT:   ret void
// CHECK: terminate.lpad:                                   ; preds = %entry
// CHECK-NEXT:   %2 = landingpad { i8*, i32 }
// CHECK-NEXT:           catch i8* null
// CHECK-NEXT:   %3 = extractvalue { i8*, i32 } %2, 0
// CHECK-NEXT:   call void @__clang_call_terminate(i8* %3) #4
// CHECK-NEXT:   unreachable

void foo2() {
    #pragma oss task
    {
        try {
            throw 1;
        } catch (int e) {
        }
    }
    #pragma oss task
    {
        int n;
        try {
            throw 1;
            n++;
        } catch (int e) {
        }
    }
}

// Each task has its own exn.slot, ehselector.slot, and all it's exception
// handling BB

// CHECK:   %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT:   %exn.slot = alloca i8*
// CHECK-NEXT:   %ehselector.slot = alloca i32
// CHECK-NEXT:   %e = alloca i32, align 4
// CHECK-NEXT:   %exception = call i8* @__cxa_allocate_exception(i64 4) #1
// CHECK-NEXT:   %1 = bitcast i8* %exception to i32*
// CHECK-NEXT:   store i32 1, i32* %1, align 16
// CHECK-NEXT:   invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
// CHECK-NEXT:           to label %invoke.cont unwind label %lpad
// CHECK: invoke.cont:                                      ; preds = %entry
// CHECK-NEXT:   br label %throw.cont
// CHECK: throw.cont:                                       ; preds = %invoke.cont
// CHECK-NEXT:   br label %try.cont
// CHECK: lpad:                                             ; preds = %entry
// CHECK-NEXT:   %2 = landingpad { i8*, i32 }
// CHECK-NEXT:           catch i8* bitcast (i8** @_ZTIi to i8*)
// CHECK-NEXT:           catch i8* null
// CHECK-NEXT:   %3 = extractvalue { i8*, i32 } %2, 0
// CHECK-NEXT:   store i8* %3, i8** %exn.slot, align 8
// CHECK-NEXT:   %4 = extractvalue { i8*, i32 } %2, 1
// CHECK-NEXT:   store i32 %4, i32* %ehselector.slot, align 4
// CHECK-NEXT:   br label %catch.dispatch
// CHECK: catch.dispatch:                                   ; preds = %lpad
// CHECK-NEXT:   %sel = load i32, i32* %ehselector.slot, align 4
// CHECK-NEXT:   %5 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #1
// CHECK-NEXT:   %matches = icmp eq i32 %sel, %5
// CHECK-NEXT:   br i1 %matches, label %catch, label %terminate.handler
// CHECK: catch:                                            ; preds = %catch.dispatch
// CHECK-NEXT:   %exn = load i8*, i8** %exn.slot, align 8
// CHECK-NEXT:   %6 = call i8* @__cxa_begin_catch(i8* %exn) #1
// CHECK-NEXT:   %7 = bitcast i8* %6 to i32*
// CHECK-NEXT:   %8 = load i32, i32* %7, align 4
// CHECK-NEXT:   store i32 %8, i32* %e, align 4
// CHECK-NEXT:   call void @__cxa_end_catch() #1
// CHECK-NEXT:   br label %try.cont
// CHECK: try.cont:                                         ; preds = %catch, %throw.cont
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %0)

// CHECK:   %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
// CHECK-NEXT:   %n = alloca i32, align 4
// CHECK-NEXT:   %exn.slot4 = alloca i8*
// CHECK-NEXT:   %ehselector.slot5 = alloca i32
// CHECK-NEXT:   %e12 = alloca i32, align 4
// CHECK-NEXT:   %exception2 = call i8* @__cxa_allocate_exception(i64 4) #1
// CHECK-NEXT:   %10 = bitcast i8* %exception2 to i32*
// CHECK-NEXT:   store i32 1, i32* %10, align 16
// CHECK-NEXT:   invoke void @__cxa_throw(i8* %exception2, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
// CHECK-NEXT:           to label %invoke.cont6 unwind label %lpad3
// CHECK: invoke.cont6:                                     ; preds = %try.cont
// CHECK-NEXT:   br label %throw.cont7
// CHECK: throw.cont7:                                      ; preds = %invoke.cont6
// CHECK-NEXT:   %11 = load i32, i32* %n, align 4
// CHECK-NEXT:   %inc = add nsw i32 %11, 1
// CHECK-NEXT:   store i32 %inc, i32* %n, align 4
// CHECK-NEXT:   br label %try.cont14
// CHECK: terminate.handler:                                ; preds = %catch.dispatch
// CHECK-NEXT:   %exn1 = load i8*, i8** %exn.slot, align 8
// CHECK-NEXT:   call void @__clang_call_terminate(i8* %exn1) #4
// CHECK-NEXT:   unreachable
// CHECK: lpad3:                                            ; preds = %try.cont
// CHECK-NEXT:   %12 = landingpad { i8*, i32 }
// CHECK-NEXT:           catch i8* bitcast (i8** @_ZTIi to i8*)
// CHECK-NEXT:           catch i8* null
// CHECK-NEXT:   %13 = extractvalue { i8*, i32 } %12, 0
// CHECK-NEXT:   store i8* %13, i8** %exn.slot4, align 8
// CHECK-NEXT:   %14 = extractvalue { i8*, i32 } %12, 1
// CHECK-NEXT:   store i32 %14, i32* %ehselector.slot5, align 4
// CHECK-NEXT:   br label %catch.dispatch8
// CHECK: catch.dispatch8:                                  ; preds = %lpad3
// CHECK-NEXT:   %sel9 = load i32, i32* %ehselector.slot5, align 4
// CHECK-NEXT:   %15 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #1
// CHECK-NEXT:   %matches10 = icmp eq i32 %sel9, %15
// CHECK-NEXT:   br i1 %matches10, label %catch11, label %terminate.handler16
// CHECK: catch11:                                          ; preds = %catch.dispatch8
// CHECK-NEXT:   %exn13 = load i8*, i8** %exn.slot4, align 8
// CHECK-NEXT:   %16 = call i8* @__cxa_begin_catch(i8* %exn13) #1
// CHECK-NEXT:   %17 = bitcast i8* %16 to i32*
// CHECK-NEXT:   %18 = load i32, i32* %17, align 4
// CHECK-NEXT:   store i32 %18, i32* %e12, align 4
// CHECK-NEXT:   call void @__cxa_end_catch() #1
// CHECK-NEXT:   br label %try.cont14
// CHECK: try.cont14:                                       ; preds = %catch11, %throw.cont7
// CHECK-NEXT:   call void @llvm.directive.region.exit(token %9)
// CHECK-NEXT:   ret void
// CHECK: terminate.handler16:                              ; preds = %catch.dispatch8
// CHECK-NEXT:   %exn15 = load i8*, i8** %exn.slot4, align 8
// CHECK-NEXT:   call void @__clang_call_terminate(i8* %exn15) #4
// CHECK-NEXT:   unreachable

