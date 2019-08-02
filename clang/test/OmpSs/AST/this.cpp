// RUN: %clang_cc1 -x c++ -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

struct S {
  int x;
  int y;
  void f(int _x) {
    #pragma oss task depend(out : x)
    {}
    #pragma oss task depend(in : this->x)
    {}
    #pragma oss task
    { x = 1; }
    #pragma oss task
    { this->x = 2; }
  }
};

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:8:13, col:37>
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:22, col:36>
// CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:35> 'int' lvalue ->x 0x{{[^ ]*}}
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:35> 'S *' implicit this
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:35> 'S *' implicit this

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:10:13, col:42>
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:22, col:41>
// CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:34, col:40> 'int' lvalue ->x 0x{{[^ ]*}}
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:34> 'S *' this
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:34> 'S *' this

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:12:13, col:21>
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <line:13:7> 'S *' implicit this

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:14:13, col:21>
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <line:15:7> 'S *' this

