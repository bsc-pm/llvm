// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

void foo(int a) {
    #pragma oss task
    {
        int b = sizeof(a);
    }
}

// CHECK: OSSTaskDirective {{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: CompoundStmt {{.*}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>

