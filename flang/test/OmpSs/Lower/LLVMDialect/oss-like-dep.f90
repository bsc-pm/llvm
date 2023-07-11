! This test checks lowering of OmpSs-2 like dependencies.

! RUN: bbc -fompss-2 %s -o - | \
! RUN:   fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

program task
    IMPLICIT NONE
    INTEGER :: I

    !$OSS TASK OUT(I)
    CONTINUE
    !$OSS END TASK
end

!LLVMIRDialect-LABEL: llvm.func @_QQmain()
!LLVMIRDialect: %[[CONSTANT_1:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT: %[[VAR_I:.*]] = llvm.alloca %[[CONSTANT_1]]
!LLVMIRDialect-NEXT: %[[DEP_0:.*]] = oss.dependency base(%[[VAR_I]] : !llvm.ptr<i32>) function(@compute.dep0) arguments(%[[VAR_I]] : !llvm.ptr<i32>) -> i32
!LLVMIRDialect-NEXT: oss.task shared(%[[VAR_I]] : !llvm.ptr<i32>) out(%[[DEP_0]] : i32)

!LLVMIRDialect-LABEL: llvm.func @compute.dep0(%arg0: !llvm.ptr<i32>) -> !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect: %0 = llvm.mlir.constant(0 : i64) : i64
!LLVMIRDialect-NEXT: %1 = llvm.mlir.constant(4 : i64) : i64
!LLVMIRDialect-NEXT: %2 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT: %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT: %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT: %5 = llvm.insertvalue %0, %4[2] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT: %6 = llvm.insertvalue %1, %5[3] : !llvm.struct<(ptr<i32>, i64, i64, i64)>
!LLVMIRDialect-NEXT: llvm.return %6 : !llvm.struct<(ptr<i32>, i64, i64, i64)>

