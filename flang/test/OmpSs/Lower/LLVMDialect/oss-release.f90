! This test checks lowering of OmpSs-2 release Directive.

! Test for program

! RUN: bbc -hlfir=false -fompss-2 %s -o - | \
! RUN:   fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

program release
    IMPLICIT NONE
    INTEGER :: I

    !$OSS RELEASE IN(I)
end

!LLVMIRDialect-LABEL: llvm.func @_QQmain()
!LLVMIRDialect:  %[[CONSTANT_1:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_I:.*]] = llvm.alloca %[[CONSTANT_1]]
!LLVMIRDialect-NEXT:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_I:.*]] : !llvm.ptr) function(@compute.dep0) arguments(%[[VAR_I:.*]] : !llvm.ptr) -> i32
!LLVMIRDialect-NEXT:  oss.release in(%[[DEP_0]] : i32)

