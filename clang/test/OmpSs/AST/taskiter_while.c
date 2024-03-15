// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

int main() {
  int i;
  #pragma oss taskiter
  while (i < 10) {
    for (int j = 0; j < 10; ++j) {}
  }
}

// CHECK: OSSTaskIterDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSFirstprivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: WhileStmt 0x{{.*}} <col{{.*}}3, line:{{.*}}:{{.*}}>
// CHECK-NEXT: BinaryOperator 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 10
// CHECK-NEXT: CompoundStmt 0x{{.*}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK-NEXT: ForStmt 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclStmt 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used j 'int' cinit
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'j' 'int'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 10
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' prefix '++'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'j' 'int'
// CHECK-NEXT: CompoundStmt 0x{{.*}} <col:{{.*}}, col:{{.*}}>

