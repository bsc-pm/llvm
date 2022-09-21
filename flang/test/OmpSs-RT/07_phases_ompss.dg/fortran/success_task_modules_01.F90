! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=config/mercurium-ompss
! compile_versions="mod0 mod1"
! test_FFLAGS_mod0="-DMOD0"
! test_FFLAGS_mod1="-DMOD1"
! </testinfo>

MODULE FOO
    CONTAINS
        !$OSS TASK INOUT(X)
        SUBROUTINE BAR(X)
            INTEGER :: X

            X = X + 1
        END SUBROUTINE BAR
END MODULE FOO

PROGRAM MAIN
    USE FOO
    IMPLICIT NONE
    INTEGER :: Y

    Y = 3

    CALL BAR(Y)
    !$OSS TASKWAIT
    IF (Y /= 4) STOP 1

END PROGRAM MAIN

