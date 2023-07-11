! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! compile_versions="mod0 mod1"
! test_FFLAGS_mod0="-DMOD0"
! test_FFLAGS_mod1="-DMOD1"
! </testinfo>

MODULE ONE
    TYPE T
        INTEGER :: A
    END TYPE T
END MODULE ONE

MODULE TWO
    CONTAINS
        !$OSS TASK INOUT(X)
        SUBROUTINE BAR(X)
            USE ONE, ONLY : T
            TYPE(T) :: X

            X % A = X % A + 1
        END SUBROUTINE BAR
END MODULE TWO

MODULE THREE
    USE ONE
END MODULE THREE

PROGRAM MAIN
    USE THREE
    USE TWO
    IMPLICIT NONE
    TYPE(T) :: Y

    Y % A = 3

    CALL BAR(Y)
    !$OSS TASKWAIT

    IF (Y % A /= 4) STOP 1
END PROGRAM MAIN
