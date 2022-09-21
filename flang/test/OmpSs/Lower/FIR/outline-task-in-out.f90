! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!$OSS TASK OUT(X)
SUBROUTINE S1(X)
  IMPLICIT NONE
  INTEGER :: X
END SUBROUTINE S1

MODULE MOO
CONTAINS
  !$OSS TASK IN(X2)
  SUBROUTINE S2(X2)
    IMPLICIT NONE
    INTEGER :: X2
  END SUBROUTINE S2
END MODULE MOO

PROGRAM MAIN
  USE MOO, ONLY : S2
  IMPLICIT NONE
  INTEGER :: Y
  CALL S1(Y)
  CALL S2(Y)
  !$OSS TASKWAIT
END PROGRAM MAIN 

!FIRDialect-LABEL: func @_QPs1(%arg0: !fir.ref<i32> {fir.bindc_name = "x"})

!FIRDialect-LABEL: func @_QMmooPs2(%arg0: !fir.ref<i32> {fir.bindc_name = "x2"})

!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[VAR_Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}

!FIRDialect:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_Y]] : !fir.ref<i32>) function(@compute.dep0) arguments(%[[VAR_Y]] : !fir.ref<i32>) -> i32
!FIRDialect-NEXT:  oss.task shared(%[[VAR_Y]] : !fir.ref<i32>) out(%[[DEP_0]] : i32)
!FIRDialect-NEXT:  fir.call @_QPs1(%[[VAR_Y]]) : (!fir.ref<i32>) -> ()
!FIRDialect-NEXT:  oss.terminator

!FIRDialect:  %[[DEP_1:.*]] = oss.dependency base(%[[VAR_Y]] : !fir.ref<i32>) function(@compute.dep1) arguments(%[[VAR_Y]] : !fir.ref<i32>) -> i32
!FIRDialect-NEXT:  oss.task shared(%[[VAR_Y]] : !fir.ref<i32>) in(%[[DEP_1]] : i32)
!FIRDialect-NEXT:  fir.call @_QMmooPs2(%[[VAR_Y]]) : (!fir.ref<i32>) -> ()
!FIRDialect-NEXT:  oss.terminator

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

!FIRDialect-LABEL: func @compute.dep1(%arg0: !fir.ref<i32>) -> (!fir.ref<i32>, i64, i64, i64)
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