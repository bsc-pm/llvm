// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

void foo1(int &rx) {}
int main() {
    int x;
    foo1(x);
}

// CHECK: CallExpr 0xefaae20 <line:4:5, col:11> 'void'
// CHECK-NEXT: ImplicitCastExpr 0xefaae08 <col:5> 'void (*)(int &)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0xefaadb8 <col:5> 'void (int &)' lvalue Function 0xefaaa98 'foo1' 'void (int &)'
// CHECK-NEXT: DeclRefExpr 0xefaad98 <col:10> 'int' lvalue Var 0xefaacd0 'x' 'int'

