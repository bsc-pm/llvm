! This test checks lowering of OmpSs-2 DepOp (outline task).

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

MODULE M
    CONTAINS
        !$OSS TASK IN(ARRAY1(99))
        SUBROUTINE S2(A, B, ARRAY, ARRAY1)
            IMPLICIT NONE
            INTEGER :: A
            INTEGER :: B
            INTEGER :: ARRAY(10)
            INTEGER :: ARRAY1(99:)

        END SUBROUTINE S2

END MODULE M

PROGRAM MAIN
    USE M
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY(10)
    INTEGER :: Y
    INTEGER :: Z
    !$OSS TASK IN(ARRAY)
    !$OSS END TASK
    !$OSS TASKWAIT

    CALL S2(X + Y, Z, ARRAY, ARRAY)
    CALL FOO()
CONTAINS
SUBROUTINE FOO()
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY(10)
    INTEGER :: Y
    INTEGER :: Z

    CALL S2(X + Y, Z, ARRAY, ARRAY)
    !$OSS TASK
    ARRAY(7) = 777
    !$OSS END TASK
    !$OSS TASKWAIT
END SUBROUTINE
END PROGRAM MAIN

!FIRDialect: TODO
