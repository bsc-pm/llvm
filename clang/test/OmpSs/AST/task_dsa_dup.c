// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

// this test checks that we only create a OSSFirstprivateClause
// for 'i' once
int main() {
    int i;
    int array[i];
    #pragma oss task in(array[i]) firstprivate(i)
    {}
}

// CHECK: OSSFirstprivateClause 0x{{.*}} <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: OSSSharedClause 0x{{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:{{.*}}> 'int[i]' lvalue Var 0x{{.*}} 'array' 'int[i]'
// CHECK-NEXT: CompoundStmt 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}>

