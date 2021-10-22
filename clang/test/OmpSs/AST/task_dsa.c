// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

int d;
void foo1() {
    int a;
    int b[5];
    int *c;
    #pragma oss task
    {a = b[0] = d = *c;}
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'd' 'int'
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int[5]' lvalue Var {{[a-z0-9]+}} 'b' 'int[5]'
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int *' lvalue Var {{[a-z0-9]+}} 'c' 'int *'

int a;
void foo2() {
  #pragma oss task
  {
    #pragma oss task firstprivate(a)
    {
      #pragma oss task
      { a++; }
    }
  }
  #pragma oss task
  {
    #pragma oss task private(a)
    {
      #pragma oss task
      { a++; }
    }
  }
  #pragma oss task firstprivate(a)
  {
    #pragma oss task shared(a)
    {
      #pragma oss task
      { a++; }
    }
  }
  #pragma oss task shared(a)
  {
    #pragma oss task private(a)
    {
      #pragma oss task
      { a++; }
    }
  }
}

// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSPrivateClause {{[a-z0-9]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSPrivateClause {{[a-z0-9]+}} <col:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <col:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-NEXT: OSSFirstprivateClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[0-9]+}}:{{[0-9]+}}> 'int' lvalue Var {{[a-z0-9]+}} 'a' 'int'

typedef int MyInt;
MyInt b;
void foo3() {
  #pragma oss task
  b++;
}
// CHECK: OSSTaskDirective {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}, col:{{[a-z0-9]+}}>
// CHECK-NEXT: OSSSharedClause {{[a-z0-9]+}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr {{[a-z0-9]+}} <line:{{[a-z0-9]+}}:{{[a-z0-9]+}}> 'MyInt':'int' lvalue Var {{[a-z0-9]+}} 'b' 'MyInt':'int'
