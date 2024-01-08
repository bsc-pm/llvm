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

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:{{.*}}> 'int' lvalue ->x 0x{{[^ ]*}}
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:{{.*}}> 'S *' implicit this
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:{{.*}}> 'S *' implicit this

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSDependClause 0x{{[^ ]*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:{{.*}}, col:{{.*}}> 'int' lvalue ->x 0x{{[^ ]*}}
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:{{.*}}> 'S *' this
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:{{.*}}> 'S *' this

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <line:{{.*}}:{{.*}}> 'S *' implicit this

// CHECK: OSSTaskDirective 0x{{[^ ]*}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: OSSSharedClause 0x{{[^ ]*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <line:15:7> 'S *' this

