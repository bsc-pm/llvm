// RUN: clang-import-test --Xcc=-fompss-2 -dump-ast -import %S/Inputs/directives.cpp -expression %s | FileCheck %s

void expr() {
    f();
}

// CHECK: OSSTaskwaitDirective
// CHECK: OSSReleaseDirective
// CHECK: OSSTaskDirective
// CHECK: OSSCriticalDirective
// CHECK: OSSTaskForDirective
// CHECK: OSSTaskIterDirective
// CHECK: OSSTaskLoopDirective
// CHECK: OSSTaskLoopForDirective
// CHECK: OSSAtomicDirective