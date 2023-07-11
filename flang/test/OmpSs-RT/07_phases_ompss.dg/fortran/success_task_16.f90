! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>

SUBROUTINE FOO(X)
    IMPLICIT NONE
    INTEGER :: X
    X = 2
END SUBROUTINE FOO

PROGRAM P
    IMPLICIT NONE
    INTERFACE
        !$OSS TASK
        SUBROUTINE FOO(X)
        IMPLICIT NONE
        INTEGER :: X
        END SUBROUTINE FOO
    END INTERFACE

    INTEGER :: X
    X = 0

    CALL FOO(X)
    !$OSS TASKWAIT

    if ( X /= 2) STOP 1
END PROGRAM P
