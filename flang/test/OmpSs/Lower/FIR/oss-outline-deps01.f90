! This test checks lowering of OmpSs-2 DepOp (outline task).

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

MODULE M
    TYPE TY
      INTEGER :: A(10)
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

!FIRDialect: TODO
