! This test checks lowering of OmpSs-2 VlaDimOp.

! Test for subroutine

! RUN: bbc -fompss-2 %s -o - | \
! RUN:   fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

subroutine task(X)
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY1(1: x + 6,1:3)
    INTEGER :: ARRAY2(1: x + 6,2:3)
    INTEGER :: ARRAY3(4: x + 6,1:3)
    INTEGER :: ARRAY4(4: x + 6,2:3)

    INTEGER :: I
    INTEGER :: J

    !$OSS TASK DO
    DO J=1,3
      DO I=1,x+6
        ARRAY1(I,J) = (I+J)
      END DO
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO
    DO J=2,3
      DO I=1,x+6
        ARRAY2(I,J) = I+J
      END DO
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO
    DO J=1,3
      DO I=4,x+6
        ARRAY3(I,J) = I+J
      END DO
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO
    DO J=2,3
      DO I=4,x+6
        ARRAY4(I,J) = I+J
      END DO
    END DO
    !$OSS END TASK DO

end subroutine

!LLVMIRDialect-LABEL: llvm.func @_QPtask(%arg0: !llvm.ptr {fir.bindc_name = "x"}) {
!LLVMIRDialect: %[[CONS_3:.*]] = llvm.mlir.constant(3 : index) : i64
!LLVMIRDialect: %[[CONS_2:.*]] = llvm.mlir.constant(2 : index) : i64
!LLVMIRDialect: %[[CONS_1:.*]] = llvm.mlir.constant(1 : index) : i64
!LLVMIRDialect: %[[CONS_4:.*]] = llvm.mlir.constant(4 : index) : i64
!LLVMIRDialect: %[[VAR_I:.*]] = llvm.alloca
!LLVMIRDialect: %[[VAR_J:.*]] = llvm.alloca

!LLVMIRDialect: %[[VAR_E1_1:.*]] = llvm.select

!LLVMIRDialect: %[[VAR_ARRAY1:.*]] = llvm.alloca

!LLVMIRDialect: %[[VAR_ARRAY2:.*]] = llvm.alloca

!LLVMIRDialect: %[[VAR_E1_2:.*]] = llvm.select

!LLVMIRDialect: %[[VAR_ARRAY3:.*]] = llvm.alloca

!LLVMIRDialect: %[[VAR_ARRAY4:.*]] = llvm.alloca

!LLVMIRDialect: %[[VLA_1:.*]] = oss.vlaDim pointer(%[[VAR_ARRAY1]] : !llvm.ptr) sizes(%[[VAR_E1_1]], %[[CONS_3]] : i64, i64) -> i32
!LLVMIRDialect: oss.task_for
!LLVMIRDialect-SAME: vlaDims(%[[VLA_1]] : i32)
!LLVMIRDialect-SAME: captures(%[[VAR_E1_1]], %[[CONS_3]] : i64, i64)

!LLVMIRDialect: %[[VLA_2:.*]] = oss.vlaDim pointer(%[[VAR_ARRAY2]] : !llvm.ptr) sizes(%[[VAR_E1_1]], %[[CONS_2]] : i64, i64) lbs(%[[CONS_1]], %[[CONS_2]] : i64, i64) -> i32
!LLVMIRDialect: oss.task_for
!LLVMIRDialect-SAME: vlaDims(%[[VLA_2]] : i32)
!LLVMIRDialect-SAME: captures(%[[VAR_E1_1]], %[[CONS_2]], %[[CONS_1]], %[[CONS_2]] : i64, i64, i64, i64)

!LLVMIRDialect: %[[VLA_3:.*]] = oss.vlaDim pointer(%[[VAR_ARRAY3]] : !llvm.ptr) sizes(%[[VAR_E1_2]], %[[CONS_3]] : i64, i64) lbs(%[[CONS_4]], %[[CONS_1]] : i64, i64) -> i32
!LLVMIRDialect: oss.task_for
!LLVMIRDialect-SAME: vlaDims(%[[VLA_3]] : i32)
!LLVMIRDialect-SAME: captures(%[[VAR_E1_2]], %[[CONS_3]], %[[CONS_4]], %[[CONS_1]] : i64, i64, i64, i64)

!LLVMIRDialect: %[[VLA_4:.*]] = oss.vlaDim pointer(%[[VAR_ARRAY4]] : !llvm.ptr) sizes(%[[VAR_E1_2]], %[[CONS_2]] : i64, i64) lbs(%[[CONS_4]], %[[CONS_2]] : i64, i64) -> i32
!LLVMIRDialect: oss.task_for
!LLVMIRDialect-SAME: vlaDims(%[[VLA_4]] : i32)
!LLVMIRDialect-SAME: captures(%[[VAR_E1_2]], %[[CONS_2]], %[[CONS_4]], %[[CONS_2]] : i64, i64, i64, i64)
