// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

int d;
int main() {
    int a;
    int b[5];
    int *c;
    #pragma oss task
    {a = b[0] = d = *c;}
}

// CHECK: OSSTaskDirective
// CHECK: OSSSharedClause 0x{{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}> 'int' lvalue Var 0x{{[a-z0-9]+}} 'a' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{[a-z0-9]+}} <col:{{[0-9]+}}> 'int [5]' lvalue Var 0x{{[a-z0-9]+}} 'b' 'int [5]'
// CHECK-NEXT: DeclRefExpr 0x{{[a-z0-9]+}} <col:{{[0-9]+}}> 'int *' lvalue Var 0x{{[a-z0-9]+}} 'c' 'int *'
// CHECK-NOT: DeclRefExpr 0x{{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}> 'int' lvalue Var 0x{{[a-z0-9]+}} 'd' 'int'
