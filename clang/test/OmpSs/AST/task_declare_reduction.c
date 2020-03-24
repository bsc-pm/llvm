// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics
void asdf(int *p, int *q) {}
#pragma oss declare reduction(foo1: int : omp_out += omp_in) initializer(omp_priv = 73 + omp_orig)
#pragma oss declare reduction(foo2: int : asdf(&omp_out, &omp_in)) initializer(omp_priv = 73 + omp_orig)
#pragma oss declare reduction(foo3: int : omp_out += omp_in) initializer(asdf(&omp_orig, &omp_priv))

// CHECK: OSSDeclareReductionDecl 0x{{.*}} <line:{{.*}}:{{.*}}> col:{{.*}} foo1 'int' combiner 0x{{.*}} initializer 0x{{.*}} omp_priv ()
// CHECK-NEXT: CompoundAssignOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_out' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_in' 'int'
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 73
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_orig' 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_in 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_out 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used omp_priv 'int' cinit
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 73
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_orig' 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_orig 'int'

// CHECK: OSSDeclareReductionDecl 0x{{.*}} <line:{{.*}}:{{.*}}> col:{{.*}} foo2 'int' combiner 0x{{.*}} initializer 0x{{.*}} omp_priv ()
// CHECK-NEXT: CallExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'void'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'void (*)(int *, int *)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'void (int *, int *)' Function 0x{{.*}} 'asdf' 'void (int *, int *)'
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_out' 'int'
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_in' 'int'
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 73
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_orig' 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_in 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_out 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used omp_priv 'int' cinit
// CHECK-NEXT: BinaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 73
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_orig' 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_orig 'int'

// CHECK: OSSDeclareReductionDecl 0x{{.*}} <line:{{.*}}:{{.*}}> col:{{.*}} foo3 'int' combiner 0x{{.*}} initializer 0x{{.*}}
// CHECK-NEXT: CompoundAssignOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_out' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_in' 'int'
// CHECK-NEXT: CallExpr 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'void'
// CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:{{.*}}> 'void (*)(int *, int *)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'void (int *, int *)' Function 0x{{.*}} 'asdf' 'void (int *, int *)'
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_orig' 'int'
// CHECK-NEXT: UnaryOperator 0x{{.*}} <col:{{.*}}, col:{{.*}}> 'int *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'omp_priv' 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_in 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_out 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_priv 'int'
// CHECK-NEXT: VarDecl 0x{{.*}} <col:{{.*}}> col:{{.*}} implicit used omp_orig 'int'

