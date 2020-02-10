// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

void foo1(int &rx) {}
int main() {
    int x;
    foo1(x);
}

// CHECK: CallExpr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'void (*)(int &)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'void (int &)' lvalue Function 0x{{.*}} 'foo1' 'void (int &)'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'x' 'int'

