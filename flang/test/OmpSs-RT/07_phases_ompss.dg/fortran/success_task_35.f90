! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
MODULE MOO
    INTEGER :: X
END MODULE MOO

SUBROUTINE SUB(Y)
    USE MOO, ONLY : X
    IMPLICIT NONE

    INTEGER :: Y

    !$OSS TASK
    CALL FOO
    !$OSS END TASK
    !$OSS TASKWAIT

    CONTAINS

        SUBROUTINE FOO
            INTEGER :: A(X+1)

            A(:) = A(:) + 1
            IF (Y+1 /= SIZE(A, DIM=1)) STOP 1
        END SUBROUTINE FOO
END SUBROUTINE SUB

PROGRAM MAIN
    USE MOO, ONLY : X
    IMPLICIT NONE

    X = 64

    CALL SUB(64)
END PROGRAM MAIN
