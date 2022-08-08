// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics
struct S {
  int a[10];
};

int global;
void foo(int i, int j) {
  struct S s;
  #pragma oss task depend(in : s.a[i+j]) depend(out: global)
  {}
}

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:10:11, col:61>
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:20, col:40>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[^ ]*}} <col:32, col:39> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:32, col:34> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:32, col:34> 'int[10]' lvalue .a 0x{{[^ ]*}}
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:32> 'struct S':'struct S' lvalue Var 0x{{[^ ]*}} 's' 'struct S':'struct S'
// CHECK-NEXT: BinaryOperator 0x{{[^ ]*}} <col:36, col:38> 'int' '+'
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:36> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:36> 'int' lvalue ParmVar 0x{{[^ ]*}} 'i' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:38> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:38> 'int' lvalue ParmVar 0x{{[^ ]*}} 'j' 'int'
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:42, col:60>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:54> 'int' lvalue Var 0x{{[^ ]*}} 'global' 'int'
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:32> 'struct S':'struct S' lvalue Var 0x{{[^ ]*}} 's' 'struct S':'struct S'
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:54> 'int' lvalue Var 0x{{[^ ]*}} 'global' 'int'
// CHECK-NEXT: OSSFirstprivateClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:36> 'int' lvalue ParmVar 0x{{[^ ]*}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:38> 'int' lvalue ParmVar 0x{{[^ ]*}} 'j' 'int'


