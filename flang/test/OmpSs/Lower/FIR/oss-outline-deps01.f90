! This test checks lowering of OmpSs-2 DepOp (outline task).

! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

MODULE M
    TYPE TY
      INTEGER :: X(10)
    END TYPE
    CONTAINS
        !$OSS TASK INOUT(T%X(1:10))
        SUBROUTINE ST(T)
            TYPE(TY) :: T
        END SUBROUTINE ST
END MODULE M

PROGRAM MAIN
    USE M
    IMPLICIT NONE

    TYPE(TY) :: T

    CALL ST(T)
    !$OSS TASKWAIT
END PROGRAM MAIN

!FIRDialect-LABEL: func.func @compute.dep0(%arg0: !fir.ref<!fir.type<_QMmTty{x:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect:   %0 = fir.field_index x, !fir.type<_QMmTty{x:!fir.array<10xi32>}>
!FIRDialect-NEXT:   %1 = fir.coordinate_of %arg0, %0 : (!fir.ref<!fir.type<_QMmTty{x:!fir.array<10xi32>}>>, !fir.field) -> !fir.ref<!fir.array<10xi32>>
!FIRDialect-NEXT:   %c10 = arith.constant 10 : index
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:   %c1_i32 = arith.constant 1 : i32
!FIRDialect-NEXT:   %2 = fir.convert %c1_i32 : (i32) -> i64
!FIRDialect-NEXT:   %c10_i32 = arith.constant 10 : i32
!FIRDialect-NEXT:   %3 = fir.convert %c10_i32 : (i32) -> i64
!FIRDialect-NEXT:   %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %4 = arith.subi %2, %c1_i64 : i64
!FIRDialect-NEXT:   %5 = arith.subi %3, %c1_i64 : i64
!FIRDialect-NEXT:   %6 = arith.addi %5, %c1_i64_0 : i64
!FIRDialect-NEXT:   %7 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT:   %8 = arith.muli %4, %c4_i64 : i64
!FIRDialect-NEXT:   %9 = arith.muli %6, %c4_i64 : i64
!FIRDialect-NEXT:   return %1, %7, %8, %9 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64
