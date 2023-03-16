! This test checks lowering of OmpSs-2 deps.

! Assumed-shape

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
PROGRAM P
  IMPLICIT NONE
  INTEGER :: ARRAY(2:10)

  CALL S(2, ARRAY)
  !$OSS TASKWAIT
CONTAINS
SUBROUTINE S(N, ARRAY)
  IMPLICIT NONE
  INTEGER :: N
  INTEGER :: ARRAY(N:)

  !$OSS TASK IN(ARRAY(4:8))
  !$OSS END TASK
END
END


!FIRDialect:  %2 = oss.dependency base(%arg1 : !fir.box<!fir.array<?xi32>>) function(@compute.dep0) arguments(%1, %arg1 : i64, !fir.box<!fir.array<?xi32>>) -> i32
!FIRDialect-NEXT:  oss.task shared(%arg1 : !fir.box<!fir.array<?xi32>>) captures(%1 : i64) in(%2 : i32) {

!FIRDialect-LABEL: func.func @compute.dep0(%arg0: i64, %arg1: !fir.box<!fir.array<?xi32>>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %0 = fir.convert %arg0 : (i64) -> index
!FIRDialect-NEXT:   %1 = fir.box_addr %arg1 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %2:3 = fir.box_dims %arg1, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %3 = fir.convert %0 : (index) -> i64
!FIRDialect-NEXT:   %4 = fir.convert %2#1 : (index) -> i64
!FIRDialect-NEXT:   %5 = arith.addi %4, %3 : i64
!FIRDialect-NEXT:   %6 = arith.subi %5, %c1_i64 : i64
!FIRDialect-NEXT:   %c4_i32 = arith.constant 4 : i32
!FIRDialect-NEXT:   %7 = fir.convert %c4_i32 : (i32) -> i64
!FIRDialect-NEXT:   %c8_i32 = arith.constant 8 : i32
!FIRDialect-NEXT:   %8 = fir.convert %c8_i32 : (i32) -> i64
!FIRDialect-NEXT:   %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %9 = arith.subi %7, %3 : i64
!FIRDialect-NEXT:   %10 = arith.subi %8, %3 : i64
!FIRDialect-NEXT:   %11 = arith.addi %10, %c1_i64_0 : i64
!FIRDialect-NEXT:   %12 = arith.muli %4, %c4_i64 : i64
!FIRDialect-NEXT:   %13 = arith.muli %9, %c4_i64 : i64
!FIRDialect-NEXT:   %14 = arith.muli %11, %c4_i64 : i64
!FIRDialect-NEXT:   return %1, %12, %13, %14 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64
