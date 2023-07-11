! This test checks lowering of OmpSs-2 like dependencies.

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program task
    IMPLICIT NONE
    INTEGER :: I

    !$OSS TASK OUT(I)
    CONTINUE
    !$OSS END TASK
end

!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
!FIRDialect-NEXT:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_I]] : !fir.ref<i32>) function(@compute.dep0) arguments(%[[VAR_I]] : !fir.ref<i32>) -> i32
!FIRDialect-NEXT:  oss.task shared(%[[VAR_I]] : !fir.ref<i32>) out(%[[DEP_0]] : i32)

!FIRDialect-LABEL: func @compute.dep0(%arg0: !fir.ref<i32>) -> (!fir.ref<i32>, i64, i64, i64)
!FIRDialect: %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT: %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT: %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT: %0 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT: %1 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT: %2 = arith.addi %1, %c1_i64_0 : i64
!FIRDialect-NEXT: %3 = arith.muli %c1_i64, %c4_i64 : i64
!FIRDialect-NEXT: %4 = arith.muli %0, %c4_i64 : i64
!FIRDialect-NEXT: %5 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT: return %arg0, %3, %4, %5 : !fir.ref<i32>, i64, i64, i64

