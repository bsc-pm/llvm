// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

template<typename T>
struct P {
  #pragma oss declare reduction(asdf : T : omp_out += omp_in) initializer(omp_priv = 0)
};

template<typename T>
struct Q {
  #pragma oss declare reduction(asdf : T : omp_out += omp_in) initializer(omp_priv = 0)
};

template<typename T>
struct S {
  #pragma oss task weakreduction(P<int>::asdf: [1]p) reduction(Q<char>::asdf: [2]q)
  void foo(int *p, char *q);
};

int main() {
  S<int> s;
  s.foo(nullptr, nullptr);
}

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK: OSSArrayShapingExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[1]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'p' 'int *'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: OSSArrayShapingExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'char[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'char *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'char *' lvalue ParmVar 0x{{.*}} 'q' 'char *'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 2
// CHECK: UnresolvedLookupExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> '<overloaded function type>' lvalue (ADL) = 'asdf' 0x{{.*}} 0x{{.*}}
// CHECK-NEXT: UnresolvedLookupExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> '<overloaded function type>' lvalue (ADL) = 'asdf' 0x{{.*}} 0x{{.*}}

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit
// CHECK: OSSArrayShapingExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int[1]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int *' lvalue ParmVar 0x{{.*}} 'p' 'int *'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: OSSArrayShapingExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'char[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'char *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'char *' lvalue ParmVar 0x{{.*}} 'q' 'char *'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 2
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} '.reduction.lhs' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'char' lvalue Var 0x{{.*}} '.reduction.lhs' 'char'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'p' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'char' lvalue Var 0x{{.*}} 'q' 'char'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue OSSDeclareReduction 0x{{.*}} 'asdf' 'int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'char' lvalue OSSDeclareReduction 0x{{.*}} 'asdf' 'char'



