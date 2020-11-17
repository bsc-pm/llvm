// RUN: %clang_cc1 -verify -fompss-2 -fexceptions -fcxx-exceptions -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

struct S {
  S(int);
  ~S();
  operator int();
};

int foo() { throw 4; }

int v[10];
void class_convertible() {
  // #pragma oss task in( { v[i], i = {S(0), S(1), S(2)} } )
  #pragma oss task in( v[S(0)], v[foo()] )
  {}
}


// CHECK: define internal %struct._depend_unpack_t @compute_dep([10 x i32]* %v)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t, align 8
// CHECK-NEXT:   %v.addr = alloca [10 x i32]*, align 8
// CHECK-NEXT:   %ref.tmp = alloca %struct.S, align 1
// CHECK-NEXT:   %exn.slot = alloca i8*, align 8
// CHECK-NEXT:   %ehselector.slot = alloca i32, align 4
// CHECK-NEXT:   store [10 x i32]* %v, [10 x i32]** %v.addr, align 8
// CHECK-NEXT:   invoke void @_ZN1SC1Ei(%struct.S* nonnull dereferenceable(1) %ref.tmp, i32{{( signext)?}} 0)
// CHECK-NEXT:           to label %invoke.cont unwind label %terminate.lpad
// CHECK: invoke.cont:                                      ; preds = %entry
// CHECK-NEXT:   %call = invoke{{( signext)?}} i32 @_ZN1ScviEv(%struct.S* nonnull dereferenceable(1) %ref.tmp)
// CHECK-NEXT:           to label %invoke.cont1 unwind label %lpad
// CHECK: invoke.cont1:                                     ; preds = %invoke.cont
// CHECK-NEXT:   %0 = sext i32 %call to i64
// CHECK-NEXT:   %1 = add i64 %0, 1
// CHECK-NEXT:   %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %v, i64 0, i64 0
// CHECK-NEXT:   %2 = mul i64 %0, 4
// CHECK-NEXT:   %3 = mul i64 %1, 4
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arraydecay, i32** %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 40, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 %2, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %3, i64* %7, align 8
// CHECK-NEXT:   call void @_ZN1SD1Ev(%struct.S* nonnull dereferenceable(1) %ref.tmp)
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t %8
// CHECK: lpad:                                             ; preds = %invoke.cont
// CHECK-NEXT:   %9 = landingpad { i8*, i32 }
// CHECK-NEXT:           catch i8* null
// CHECK-NEXT:   %10 = extractvalue { i8*, i32 } %9, 0
// CHECK-NEXT:   store i8* %10, i8** %exn.slot, align 8
// CHECK-NEXT:   %11 = extractvalue { i8*, i32 } %9, 1
// CHECK-NEXT:   store i32 %11, i32* %ehselector.slot, align 4
// CHECK-NEXT:   call void @_ZN1SD1Ev(%struct.S* nonnull dereferenceable(1) %ref.tmp)
// CHECK-NEXT:   br label %terminate.handler
// CHECK: terminate.lpad:                                   ; preds = %entry
// CHECK-NEXT:   %12 = landingpad { i8*, i32 }
// CHECK-NEXT:           catch i8* null
// CHECK-NEXT:   %13 = extractvalue { i8*, i32 } %12, 0
// CHECK-NEXT:   call void @__clang_call_terminate(i8* %13)
// CHECK-NEXT:   unreachable
// CHECK: terminate.handler:                                ; preds = %lpad
// CHECK-NEXT:   %exn = load i8*, i8** %exn.slot, align 8
// CHECK-NEXT:   call void @__clang_call_terminate(i8* %exn)
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }

// CHECK: define internal %struct._depend_unpack_t.0 @compute_dep.1([10 x i32]* %v)
// CHECK: entry:
// CHECK-NEXT:   %retval = alloca %struct._depend_unpack_t.0, align 8
// CHECK-NEXT:   %v.addr = alloca [10 x i32]*, align 8
// CHECK-NEXT:   store [10 x i32]* %v, [10 x i32]** %v.addr, align 8
// CHECK-NEXT:   %call = invoke{{( signext)?}} i32 @_Z3foov()
// CHECK-NEXT:           to label %invoke.cont unwind label %terminate.lpad
// CHECK: invoke.cont:                                      ; preds = %entry
// CHECK-NEXT:   %0 = sext i32 %call to i64
// CHECK-NEXT:   %1 = add i64 %0, 1
// CHECK-NEXT:   %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %v, i64 0, i64 0
// CHECK-NEXT:   %2 = mul i64 %0, 4
// CHECK-NEXT:   %3 = mul i64 %1, 4
// CHECK-NEXT:   %4 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 0
// CHECK-NEXT:   store i32* %arraydecay, i32** %4, align 8
// CHECK-NEXT:   %5 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 1
// CHECK-NEXT:   store i64 40, i64* %5, align 8
// CHECK-NEXT:   %6 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 2
// CHECK-NEXT:   store i64 %2, i64* %6, align 8
// CHECK-NEXT:   %7 = getelementptr inbounds %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, i32 0, i32 3
// CHECK-NEXT:   store i64 %3, i64* %7, align 8
// CHECK-NEXT:   %8 = load %struct._depend_unpack_t.0, %struct._depend_unpack_t.0* %retval, align 8
// CHECK-NEXT:   ret %struct._depend_unpack_t.0 %8
// CHECK: terminate.lpad:                                   ; preds = %entry
// CHECK-NEXT:   %9 = landingpad { i8*, i32 }
// CHECK-NEXT:           catch i8* null
// CHECK-NEXT:   %10 = extractvalue { i8*, i32 } %9, 0
// CHECK-NEXT:   call void @__clang_call_terminate(i8* %10)
// CHECK-NEXT:   unreachable
// CHECK-NEXT: }
