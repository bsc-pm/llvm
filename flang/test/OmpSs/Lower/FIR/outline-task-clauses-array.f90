! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!$OSS TASK INOUT(X)
SUBROUTINE S1(X)
  IMPLICIT NONE
  INTEGER :: X(10)
END SUBROUTINE S1

MODULE MOO
CONTAINS
  !$OSS TASK INOUT(X)
  SUBROUTINE S2(X)
    IMPLICIT NONE
    INTEGER :: X(10)
  END SUBROUTINE S2
END MODULE MOO

PROGRAM MAIN
  USE MOO, ONLY : S2
  IMPLICIT NONE
  INTEGER :: X(10)
  CALL S1(X)
  CALL S2(X)
  !$OSS TASKWAIT
END PROGRAM MAIN 

!FIRDialect-LABEL: func @_QPs1(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "x"})

!FIRDialect-LABEL: func @_QMmooPs2(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "x"})

!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[VAR_X:.*]] = fir.address_of(@_QFEx) : !fir.ref<!fir.array<10xi32>>

!FIRDialect:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_X]] : !fir.ref<!fir.array<10xi32>>) function(@compute.dep0) arguments(%[[VAR_X]] : !fir.ref<!fir.array<10xi32>>) -> i32
!FIRDialect-NEXT:  oss.task shared(%[[VAR_X]] : !fir.ref<!fir.array<10xi32>>) inout(%[[DEP_0]] : i32)
!FIRDialect-NEXT:  fir.call @_QPs1(%[[VAR_X]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
!FIRDialect-NEXT:  oss.terminator

!FIRDialect:  %[[DEP_1:.*]] = oss.dependency base(%[[VAR_X]] : !fir.ref<!fir.array<10xi32>>) function(@compute.dep1) arguments(%[[VAR_X]] : !fir.ref<!fir.array<10xi32>>) -> i32
!FIRDialect-NEXT:  oss.task shared(%[[VAR_X]] : !fir.ref<!fir.array<10xi32>>) inout(%[[DEP_1]] : i32)
!FIRDialect-NEXT:  fir.call @_QMmooPs2(%[[VAR_X]]) fastmath<contract> : (!fir.ref<!fir.array<10xi32>>) -> ()
!FIRDialect-NEXT:  oss.terminator

!FIRDialect-LABEL: func @compute.dep0(%arg0: !fir.ref<!fir.array<10xi32>>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect: %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT: %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT: %c10_i64_0 = arith.constant 10 : i64
!FIRDialect-NEXT: %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT: %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT: %0 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT: %1 = arith.subi %c10_i64_0, %c1_i64 : i64
!FIRDialect-NEXT: %2 = arith.addi %1, %c1_i64_1 : i64
!FIRDialect-NEXT: %3 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT: %4 = arith.muli %0, %c4_i64 : i64
!FIRDialect-NEXT: %5 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT: return %arg0, %3, %4, %5 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64

!FIRDialect-LABEL: func @compute.dep1(%arg0: !fir.ref<!fir.array<10xi32>>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect: %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT: %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT: %c10_i64_0 = arith.constant 10 : i64
!FIRDialect-NEXT: %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT: %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT: %0 = arith.subi %c1_i64, %c1_i64 : i64
!FIRDialect-NEXT: %1 = arith.subi %c10_i64_0, %c1_i64 : i64
!FIRDialect-NEXT: %2 = arith.addi %1, %c1_i64_1 : i64
!FIRDialect-NEXT: %3 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT: %4 = arith.muli %0, %c4_i64 : i64
!FIRDialect-NEXT: %5 = arith.muli %2, %c4_i64 : i64
!FIRDialect-NEXT: return %arg0, %3, %4, %5 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64
