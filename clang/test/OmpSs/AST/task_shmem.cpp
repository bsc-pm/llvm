// RUN: %clang_cc1 -verify -fompss-2 -ast-dump -ferror-limit 100 %s | FileCheck %s
// expected-no-diagnostics

#pragma oss task device(cuda) ndrange(1, 1, 1) shmem(0)
template<int N>
void foo();
#pragma oss task device(opencl) ndrange(1, 1, 1) shmem(1)
template<typename T>
void foo1(T *x);

void bad() {
    int x;
    foo<1>();
    foo1(&x);
}

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Cuda
// CHECK: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK: ConstantExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Cuda
// CHECK: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 0
// CHECK: ConstantExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Opencl
// CHECK: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK: ConstantExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

// CHECK: OSSTaskDeclAttr 0x{{.*}} <line:{{.*}}:{{.*}}, col:{{.*}}> Implicit Opencl
// CHECK: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK: ConstantExpr 0x{{.*}} <col:{{.*}}> 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: IntegerLiteral 0x{{.*}} <col:{{.*}}> 'int' 1

