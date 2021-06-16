// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

void bar(int n) {
    int vla[n];
    // #pragma oss task for collapse(1)
    // for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop collapse(1)
    for (int i = 0; i < 10; ++i) {}
    // #pragma oss taskloop for collapse(1)
    // for (int i = 0; i < 10; ++i) {}
}

// CHECK: OSSTaskLoopDirective 0x{{.*}} <line:{{.*}}:{{.*}}, <invalid sloc>> ompss-2
// CHECK-NEXT: OSSCollapseClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: ConstantExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
