// RUN: %clang_cc1 -verify -fompss-2 -fompss-2=libnodes -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
  #pragma oss task
  {
    #pragma oss task for
    for (int i = 0; i < 10; ++i) { }
    #pragma oss taskloop
    for (int i = 0; i < 10; ++i) { }
    #pragma oss taskloop for
    for (int i = 0; i < 10; ++i) { }
    #pragma oss taskiter
    for (int i = 0; i < 10; ++i) { }
    int j = 0;
    #pragma oss taskiter
    while (j > 10) {}
  }
}

// Data-sharing of induction variables

// CHECK: OSSTaskDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: CompoundStmt 0x{{.*}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: OSSTaskForDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSPrivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK: OSSTaskLoopDirective 0x{{.*}} <line:{{.*}}:{{.*}}, <invalid sloc>> ompss-2
// CHECK-NEXT: OSSPrivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK: OSSTaskLoopForDirective 0x{{.*}} <line:{{.*}}:{{.*}}, <invalid sloc>> ompss-2
// CHECK-NEXT: OSSPrivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK: OSSTaskIterDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSFirstprivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <line:{{.*}}:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK: OSSTaskIterDirective 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> ompss-2
// CHECK-NEXT: OSSFirstprivateClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <line:{{.*}}:{{.*}}2> 'int' lvalue Var 0x{{.*}} 'j' 'int'

