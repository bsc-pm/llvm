! This test checks lowering of OmpSs-2 DepOp
! derived type (outline task).

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

MODULE M
    TYPE TY
      INTEGER :: A(10)
    END TYPE
END MODULE M

!$OSS TASK INOUT(T%A(4))
SUBROUTINE ST(T)
    USE M
    TYPE(TY) :: T
END SUBROUTINE ST

PROGRAM MAIN
    USE M
    IMPLICIT NONE

    TYPE(TY) :: T

    CALL ST(T)
    !$OSS TASKWAIT
END PROGRAM MAIN

!FIRDialect-LABEL: func.func @_QQmain()
!FIRDialect: %[[VAR_T:.*]] = fir.address_of(@_QFEt) : !fir.ref<!fir.type<_QMmTty{a:!fir.array<10xi32>}>>
!FIRDialect-NEXT: %[[DEP_0:.*]] = oss.dependency base(%[[VAR_T]] : !fir.ref<!fir.type<_QMmTty{a:!fir.array<10xi32>}>>) function(@compute.dep0) arguments(%[[VAR_T]] : !fir.ref<!fir.type<_QMmTty{a:!fir.array<10xi32>}>>) -> i32
!FIRDialect-NEXT: oss.task
!FIRDialect-SAME: shared(%[[VAR_T]] : !fir.ref<!fir.type<_QMmTty{a:!fir.array<10xi32>}>>)
!FIRDialect-SAME: inout(%[[DEP_0]] : i32)

!FIRDialect-LABEL: func.func @compute.dep0(%arg0: !fir.ref<!fir.type<_QMmTty{a:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.array<10xi32>>, i64, i64, i64)
!FIRDialect:   %0 = fir.field_index a, !fir.type<_QMmTty{a:!fir.array<10xi32>}>
!FIRDialect-NEXT:   %1 = fir.coordinate_of %arg0, %0 : (!fir.ref<!fir.type<_QMmTty{a:!fir.array<10xi32>}>>, !fir.field) -> !fir.ref<!fir.array<10xi32>>
!FIRDialect-NEXT:   %c10 = arith.constant 10 : index
!FIRDialect-NEXT:   %c1_i64 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c10_i64 = arith.constant 10 : i64
!FIRDialect-NEXT:   %c4_i32 = arith.constant 4 : i32
!FIRDialect-NEXT:   %2 = fir.convert %c4_i32 : (i32) -> i64
!FIRDialect-NEXT:   %c1_i64_0 = arith.constant 1 : i64
!FIRDialect-NEXT:   %c4_i64 = arith.constant 4 : i64
!FIRDialect-NEXT:   %3 = arith.subi %2, %c1_i64 : i64
!FIRDialect-NEXT:   %4 = arith.subi %2, %c1_i64 : i64
!FIRDialect-NEXT:   %5 = arith.addi %4, %c1_i64_0 : i64
!FIRDialect-NEXT:   %6 = arith.muli %c10_i64, %c4_i64 : i64
!FIRDialect-NEXT:   %7 = arith.muli %3, %c4_i64 : i64
!FIRDialect-NEXT:   %8 = arith.muli %5, %c4_i64 : i64
!FIRDialect-NEXT:   return %1, %6, %7, %8 : !fir.ref<!fir.array<10xi32>>, i64, i64, i64
