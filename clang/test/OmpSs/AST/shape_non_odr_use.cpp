// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

struct S {
    static constexpr int N = 4;
};

void foo(int *p) {
    #pragma oss task in([S::N]p)
    {}
}

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSDependClause 0x{{.*}} <col:{{.*}}, col:{{.*}}> <oss syntax>
// CHECK-NEXT: OSSArrayShapingExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[4]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'p' 'int *'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'const int' lvalue Var 0x{{.*}} 'N' 'const int' non_odr_use_constant
// CHECK-NEXT: OSSFirstprivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'p' 'int *'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'const int' lvalue Var 0x{{.*}} 'N' 'const int' non_odr_use_constant
