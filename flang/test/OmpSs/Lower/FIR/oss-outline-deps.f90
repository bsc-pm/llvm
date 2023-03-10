! This test checks lowering of OmpSs-2 DepOp (outline task).

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
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
!FIRDialect-NEXT:   %c1 = arith.constant 1 : index
!FIRDialect-NEXT:   %3 = fir.convert %0 : (index) -> i64
!FIRDialect-NEXT:   %4 = fir.convert %2#1 : (index) -> i64
!FIRDialect-NEXT:   %5 = arith.addi %4, %3 : i64
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %6 = arith.subi %5, %c1_i64 : i64
!FIRDialect-NEXT:   %7 = arith.subi %6, %3 : i64
!FIRDialect-NEXT:   %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:   %8 = arith.addi %7, %c1_i64_0 : i64
!FIRDialect-NEXT:   %c0_i64 = arith.constant 0 : i64
!FIRDialect-NEXT:   %9 = arith.cmpi sgt, %8, %c0_i64 : i64
!FIRDialect-NEXT:   %10 = arith.select %9, %8, %c0_i64 : i64
!FIRDialect-NEXT:   %c99_i32 = arith.constant 99 : i32
!FIRDialect-NEXT:   %11 = fir.convert %c99_i32 : (i32) -> i64
!FIRDialect-NEXT:   %c1_i64_1 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %12 = arith.subi %11, %3 : i64
!FIRDialect-NEXT:   %13 = arith.subi %11, %3 : i64
!FIRDialect-NEXT:   %14 = arith.addi %13, %c1_i64_1 : i64
!FIRDialect-NEXT:   %15 = arith.muli %10, %c4_i64 : i64
!FIRDialect-NEXT:   %16 = arith.muli %12, %c4_i64 : i64
!FIRDialect-NEXT:   %17 = arith.muli %14, %c4_i64 : i64
!FIRDialect-NEXT:   return %1, %15, %16, %17 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64

