// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(fpga) num_instances(2) onto(0x300000000) copy_deps
void foo0() {
    #pragma HLS
}
#pragma oss task device(fpga) period(12) affinity(1234)
void foo1() {
    #pragma HLS TOK
}

#pragma oss task device(fpga) copy_in([1000]i) copy_out([1000]o) copy_inout([1000]io)
void foo2(int *i, int *o, int *io) {
    #pragma HLS TOK=VAL
}

// CHECK:TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK:|-FunctionDecl {{.*}} <{{.*}}task_fpga.c:5:1, line:7:1> line:5:6 foo0 'void ()'
// CHECK-NEXT:| |-CompoundStmt {{.*}} <col:13, line:7:1>
// CHECK-NEXT:| | `-HlsDirective {{.*}} <line:6:13, col:16>HLS
// CHECK-NEXT:| `-OSSTaskDeclAttr {{.*}} <line:4:9, col:75> Implicit CopyDeps Fpga
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-ConstantExpr {{.*}} <col:45> 'int'
// CHECK-NEXT:|   | |-value: Int 2
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} <col:45> 'int' 2
// CHECK-NEXT:|   |-ConstantExpr {{.*}} <col:53> 'long'
// CHECK-NEXT:|   | |-value: Int 12884901888
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} <col:53> 'long' 12884901888
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   `-<<<NULL>>>
// CHECK-NEXT:|-FunctionDecl {{.*}} <line:9:1, line:11:1> line:9:6 foo1 'void ()'
// CHECK-NEXT:| |-CompoundStmt {{.*}} <col:13, line:11:1>
// CHECK-NEXT:| | `-HlsDirective {{.*}} <line:10:13, col:20>HLS TOK
// CHECK-NEXT:| `-OSSTaskDeclAttr {{.*}} <line:8:9, col:56> Implicit Fpga
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-<<<NULL>>>
// CHECK-NEXT:|   |-ConstantExpr {{.*}} <col:38> 'int'
// CHECK-NEXT:|   | |-value: Int 12
// CHECK-NEXT:|   | `-IntegerLiteral {{.*}} <col:38> 'int' 12
// CHECK-NEXT:|   `-ConstantExpr {{.*}} <col:51> 'int'
// CHECK-NEXT:|     |-value: Int 1234
// CHECK-NEXT:|     `-IntegerLiteral {{.*}} <col:51> 'int' 1234
// CHECK-NEXT:`-FunctionDecl {{.*}} <line:14:1, line:16:1> line:14:6 foo2 'void (int *, int *, int *)'
// CHECK-NEXT:  |-ParmVarDecl {{.*}} <col:11, col:16> col:16 used i 'int *'
// CHECK-NEXT:  |-ParmVarDecl {{.*}} <col:19, col:24> col:24 used o 'int *'
// CHECK-NEXT:  |-ParmVarDecl {{.*}} <col:27, col:32> col:32 used io 'int *'
// CHECK-NEXT:  |-CompoundStmt {{.*}} <col:36, line:16:1>
// CHECK-NEXT:  | `-HlsDirective {{.*}} <line:15:13, col:24>HLS TOK=VAL
// CHECK-NEXT:  `-OSSTaskDeclAttr {{.*}} <line:13:9, col:86> Implicit Fpga
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-<<<NULL>>>
// CHECK-NEXT:    |-OSSArrayShapingExpr {{.*}} <col:39, col:45> 'int[1000]' lvalue
// CHECK-NEXT:    | |-ImplicitCastExpr {{.*}} <col:45> 'int *' <LValueToRValue>
// CHECK-NEXT:    | | `-DeclRefExpr {{.*}} <col:45> 'int *' lvalue ParmVar {{.*}} 'i' 'int *'
// CHECK-NEXT:    | `-ConstantExpr {{.*}} <col:40> 'int'
// CHECK-NEXT:    |   |-value: Int 1000
// CHECK-NEXT:    |   `-IntegerLiteral {{.*}} <col:40> 'int' 1000
// CHECK-NEXT:    |-OSSArrayShapingExpr {{.*}} <col:57, col:63> 'int[1000]' lvalue
// CHECK-NEXT:    | |-ImplicitCastExpr {{.*}} <col:63> 'int *' <LValueToRValue>
// CHECK-NEXT:    | | `-DeclRefExpr {{.*}} <col:63> 'int *' lvalue ParmVar {{.*}} 'o' 'int *'
// CHECK-NEXT:    | `-ConstantExpr {{.*}} <col:58> 'int'
// CHECK-NEXT:    |   |-value: Int 1000
// CHECK-NEXT:    |   `-IntegerLiteral {{.*}} <col:58> 'int' 1000
// CHECK-NEXT:    `-OSSArrayShapingExpr {{.*}} <col:77, col:83> 'int[1000]' lvalue
// CHECK-NEXT:      |-ImplicitCastExpr {{.*}} <col:83> 'int *' <LValueToRValue>
// CHECK-NEXT:      | `-DeclRefExpr {{.*}} <col:83> 'int *' lvalue ParmVar {{.*}} 'io' 'int *'
// CHECK-NEXT:      `-ConstantExpr {{.*}} <col:78> 'int'
// CHECK-NEXT:        |-value: Int 1000
// CHECK-NEXT:        `-IntegerLiteral {{.*}} <col:78> 'int' 1000
