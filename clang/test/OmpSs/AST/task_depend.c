// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

int main() {
    int a;
    int b[5];
    int *c;
    #pragma oss task depend(in : a, b[0], *c)
    { a = b[0] = *c; }
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'b' 'int [5]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' 0
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'c' 'int *'
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'b' 'int [5]'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'c' 'int *'

int *p;
int foo() {
  #pragma oss task depend(inout: *p)
  {
    #pragma oss task
    {
      *p = 0;
    }
  }
  return 0;
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: UnaryOperator {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'
// CHECK-NEXT: CompoundStmt {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, line:{{[a-z0-9]+}}:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'p' 'int *'

int array[5];
int foo1() {
  #pragma oss task depend(inout: array[2])
  {
    #pragma oss task
    {
      array[2] = 0;
    }
  }
  return 0;
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' 2
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'
// CHECK-NEXT: CompoundStmt {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, line:{{[a-z0-9]+}}:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'

int foo2() {
  int array[5];
  #pragma oss task depend(inout: array[2])
  {
    #pragma oss task
    {
      array[2] = 0;
    }
  }
  return 0;
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'
// CHECK-NEXT: IntegerLiteral {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' 2
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'
// CHECK-NEXT: CompoundStmt {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, line:{{[a-z0-9]+}}:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}> 'int [5]' lvalue Var {{[a-z0-9]+}} 'array' 'int [5]'

void foo3() {
    int array[10];
    int i;
    struct A {
        int x;
    } a;
    int index[10];
    #pragma oss task depend(in: array[i])
    #pragma oss task depend(in: array[i + a.x])
    #pragma oss task depend(in: array[a.x])
    #pragma oss task depend(in: array[index[i]])
    {}
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'array' 'int [10]'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'i' 'int'
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'array' 'int [10]'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'i' 'int'
// CHECK-NEXT: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'array' 'int [10]'
// CHECK-NEXT: BinaryOperator {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' '+'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'i' 'int'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue .x {{[a-z0-9]+}}
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'struct A':'struct A' lvalue Var {{[a-z0-9]+}} 'a' 'struct A':'struct A'
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'array' 'int [10]'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'i' 'int'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'struct A':'struct A' lvalue Var {{[a-z0-9]+}} 'a' 'struct A':'struct A'
// CHECK-NEXT: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'array' 'int [10]'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue .x {{[a-z0-9]+}}
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'struct A':'struct A' lvalue Var {{[a-z0-9]+}} 'a' 'struct A':'struct A'
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'array' 'int [10]'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'struct A':'struct A' lvalue Var {{[a-z0-9]+}} 'a' 'struct A':'struct A'
// CHECK-NEXT: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSDependClause {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'array' 'int [10]'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}> 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'index' 'int [10]'
// CHECK-NEXT: ImplicitCastExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'i' 'int'
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'array' 'int [10]'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int [10]' lvalue Var {{[a-z0-9]+}} 'index' 'int [10]'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[a-z0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'i' 'int'
// CHECK-NEXT: CompoundStmt {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
void foo4() {
    int array[10][10];
    int a;
    #pragma oss task depend(out : array[0 : 5])
    {}
    #pragma oss task depend(out : array[ : a])
    {}
    #pragma oss task depend(out : array[ : ])
    {}
    #pragma oss task depend(out : array[ : ][ : ])
    {}
}

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:175:13, col:48>
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:22, col:47>
// CHECK-NEXT: OSSArraySectionExpr 0x{{[^ ]*}} <col:35, col:46> '<OmpSs-2 array section type>' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:35> 'int (*)[10]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:35> 'int [10][10]' lvalue Var 0x{{[^ ]*}} 'array' 'int [10][10]'
// CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:41> 'int' 0
// CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:45> 'int' 5
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:35> 'int [10][10]' lvalue Var 0x{{[^ ]*}} 'array' 'int [10][10]'
// CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <line:176:5, col:6>
// CHECK-NEXT: OSSTaskDirective 0x{{[^ ]*}} <line:177:13, col:47>
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:22, col:46>
// CHECK-NEXT: OSSArraySectionExpr 0x{{[^ ]*}} <col:35, col:45> '<OmpSs-2 array section type>' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:35> 'int (*)[10]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:35> 'int [10][10]' lvalue Var 0x{{[^ ]*}} 'array' 'int [10][10]'
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:44> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:44> 'int' lvalue Var 0x{{[^ ]*}} 'a' 'int'
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:35> 'int [10][10]' lvalue Var 0x{{[^ ]*}} 'array' 'int [10][10]'
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:44> 'int' lvalue Var 0x{{[^ ]*}} 'a' 'int'
// CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <line:178:5, col:6>
// CHECK-NEXT: OSSTaskDirective 0x{{[^ ]*}} <line:179:13, col:46>
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:22, col:45>
// CHECK-NEXT: OSSArraySectionExpr 0x{{[^ ]*}} <col:35, col:44> '<OmpSs-2 array section type>' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:35> 'int (*)[10]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:35> 'int [10][10]' lvalue Var 0x{{[^ ]*}} 'array' 'int [10][10]'
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:35> 'int [10][10]' lvalue Var 0x{{[^ ]*}} 'array' 'int [10][10]'
// CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <line:180:5, col:6>
// CHECK-NEXT: OSSTaskDirective 0x{{[^ ]*}} <line:181:13, col:51>
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:22, col:50>
// CHECK-NEXT: OSSArraySectionExpr 0x{{[^ ]*}} <col:35, col:49> '<OmpSs-2 array section type>' lvalue
// CHECK-NEXT: OSSArraySectionExpr 0x{{[^ ]*}} <col:35, col:44> '<OmpSs-2 array section type>' lvalue
// CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:35> 'int (*)[10]' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:35> 'int [10][10]' lvalue Var 0x{{[^ ]*}} 'array' 'int [10][10]'
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: <<<NULL>>>
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:35> 'int [10][10]' lvalue Var 0x{{[^ ]*}} 'array' 'int [10][10]'
// CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <line:182:5, col:6>

