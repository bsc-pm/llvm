! This test checks lowering of OmpSs-2 DepOp (outline task).

! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

MODULE M
    CONTAINS
        !$OSS TASK IN(ARRAY(99))
        SUBROUTINE S2(ARRAY)
            IMPLICIT NONE
            INTEGER :: ARRAY(99:)
        END SUBROUTINE S2
END MODULE M

PROGRAM MAIN
    USE M
    IMPLICIT NONE
    INTEGER :: ARRAY(10)

    CALL S2(ARRAY)
END PROGRAM MAIN

!FIRDialect-LABEL: func.func @compute.dep0(%arg0: !fir.box<!fir.array<10xi32>>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect:   %c99_i64 = arith.constant 99 : i64
!FIRDialect-NEXT:   %0 = fir.convert %c99_i64 : (i64) -> index
!FIRDialect-NEXT:   %1 = fir.box_addr %arg0 : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
!FIRDialect-NEXT:   %c0 = arith.constant 0 : index
!FIRDialect-NEXT:   %2:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<10xi32>>, index) -> (index, index, index)
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %3 = fir.convert %0 : (index) -> i64
!FIRDialect-NEXT:   %4 = fir.convert %2#1 : (index) -> i64
!FIRDialect-NEXT:   %5 = arith.addi %4, %3 : i64
!FIRDialect-NEXT:   %6 = arith.subi %5, %c1_i64 : i64
!FIRDialect-NEXT:   %c99_i32 = arith.constant 99 : i32
!FIRDialect-NEXT:   %7 = fir.convert %c99_i32 : (i32) -> i64
!FIRDialect-NEXT:   %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %8 = arith.subi %7, %3 : i64
!FIRDialect-NEXT:   %9 = arith.subi %7, %3 : i64
!FIRDialect-NEXT:   %10 = arith.addi %9, %c1_i64_0 : i64
!FIRDialect-NEXT:   %11 = arith.muli %4, %c4_i64 : i64
!FIRDialect-NEXT:   %12 = arith.muli %8, %c4_i64 : i64
!FIRDialect-NEXT:   %13 = arith.muli %10, %c4_i64 : i64
!FIRDialect-NEXT:   return %1, %11, %12, %13 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64

