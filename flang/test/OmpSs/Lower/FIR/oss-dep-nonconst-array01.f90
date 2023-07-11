! This test checks lowering of OmpSs-2 deps.

! explicit-shape non constant shape

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

PROGRAM P
  IMPLICIT NONE
  INTEGER :: ARRAY(2:10)

  CALL S(2)
  !$OSS TASKWAIT
CONTAINS
SUBROUTINE S(N)
  IMPLICIT NONE
  INTEGER :: N
  INTEGER :: ARRAY(N:10)

  !$OSS TASK IN(ARRAY(4:8))
  !$OSS END TASK
END
END

!FIRDialect:  %9 = oss.vlaDim pointer(%8 : !fir.ref<!fir.array<?xi32>>) sizes(%7 : index) lbs(%2 : index) -> i32
!FIRDialect-NEXT:  %10 = fir.load %arg0 : !fir.ref<i32>
!FIRDialect-NEXT:  %11 = fir.convert %10 : (i32) -> i64
!FIRDialect-NEXT:  %12 = oss.dependency base(%8 : !fir.ref<!fir.array<?xi32>>) function(@compute.dep0) arguments(%11, %8 : i64, !fir.ref<!fir.array<?xi32>>) -> i32
!FIRDialect-NEXT:  oss.task shared(%8 : !fir.ref<!fir.array<?xi32>>) vlaDims(%9 : i32) captures(%7, %2, %11 : index, index, i64) in(%12 : i32) {

!FIRDialect-LABEL: func.func @compute.dep0(%arg0: i64, %arg1: !fir.ref<!fir.array<?xi32>>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64)
!FIRDialect:   %0 = fir.convert %arg0 : (i64) -> index
!FIRDialect-NEXT:   %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:   %1 = fir.convert %c10_i64 : (i64) -> index
!FIRDialect-NEXT:   %2 = arith.subi %1, %0 : index
!FIRDialect-NEXT:   %c1 = arith.constant 1 : index
!FIRDialect-NEXT:   %3 = arith.addi %2, %c1 : index
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %4 = arith.cmpi sgt, %3, %c0 : index
!FIRDialect-NEXT:   %5 = arith.select %4, %3, %c0 : index
!FIRDialect-NEXT:   %c10_i64_0 = arith.constant 10 : i64
!FIRDialect-NEXT:   %6 = arith.subi %c10_i64_0, %arg0 : i64
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %7 = arith.addi %6, %c1_i64 : i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %8 = arith.cmpi sgt, %7, %c0_i64 : i64
!FIRDialect-NEXT:   %9 = arith.select %8, %7, %c0_i64 : i64
!FIRDialect-NEXT:   %c4_i32 = arith.constant 4 : i32
!FIRDialect-NEXT:   %10 = fir.convert %c4_i32 : (i32) -> i64
!FIRDialect-NEXT:   %c8_i32 = arith.constant 8 : i32
!FIRDialect-NEXT:   %11 = fir.convert %c8_i32 : (i32) -> i64
!FIRDialect-NEXT:   %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %12 = arith.subi %10, %arg0 : i64
!FIRDialect-NEXT:   %13 = arith.subi %11, %arg0 : i64
!FIRDialect-NEXT:   %14 = arith.addi %13, %c1_i64_1 : i64
!FIRDialect-NEXT:   %15 = arith.muli %9, %c4_i64 : i64
!FIRDialect-NEXT:   %16 = arith.muli %12, %c4_i64 : i64
!FIRDialect-NEXT:   %17 = arith.muli %14, %c4_i64 : i64
!FIRDialect-NEXT:   return %arg1, %15, %16, %17 : !fir.ref<!fir.array<?xi32>>, i64, i64, i64


