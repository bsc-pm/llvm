// RUN: %clang_cc1 -triple x86_64 -verify -fompss-2 -disable-llvm-passes -std=c++20 %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#include "Inputs/std-coroutine.h"

struct oss_coroutine {
  struct promise_type {
          oss_coroutine get_return_object() { return oss_coroutine{std::coroutine_handle<promise_type>::from_promise(*this)}; }
          std::suspend_never initial_suspend() { return {}; }
          std::suspend_always final_suspend() noexcept { return {}; }
          void return_void() {}
          void unhandled_exception() {}
  };
  std::coroutine_handle<> handle;
};

#pragma oss task
oss_coroutine kk2() {
  co_return;
}

int main() {
  kk2();
}

// CHECK:    [[TMP3:%.*]] = call i64 @llvm.coro.size.i64(), !dbg [[DBG10:![0-9]+]]
// CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.coro.size.storage.i64.i64(i64 [[TMP3]], ptr @[[GLOB0:[0-9]+]]), !dbg [[DBG10]]

// CHECK-LABEL: @main(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[HANDLE:%.*]] = alloca [[STRUCT_OSS_COROUTINE:%.*]], align 1
// CHECK-NEXT:    store ptr null, ptr [[HANDLE]], align 1, !dbg [[DBG41:![0-9]+]]
// CHECK-NEXT:    [[TMP0:%.*]] = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"),
// CHECK-SAME:    "QUAL.OSS.CORO.HANDLE"(ptr [[HANDLE]]),
// CHECK-SAME:    "QUAL.OSS.CORO.SIZE.STORE"(ptr [[GLOB1:@.*]]),
// CHECK-SAME:    "QUAL.OSS.FIRSTPRIVATE"(ptr [[HANDLE]], [[STRUCT_OSS_COROUTINE]] undef) ], !dbg [[DBG41]]
// CHECK-NEXT:    [[UNDEF_AGG_TMP:%.*]] = alloca [[STRUCT_OSS_COROUTINE]], align 1
// CHECK-NEXT:    call void @_Z3kk2v(), !dbg [[DBG41]]
// CHECK-NEXT:    call void @llvm.directive.region.exit(token [[TMP0]]), !dbg [[DBG41]]
// CHECK-NEXT:    ret i32 0, !dbg [[DBG42:![0-9]+]]
//
