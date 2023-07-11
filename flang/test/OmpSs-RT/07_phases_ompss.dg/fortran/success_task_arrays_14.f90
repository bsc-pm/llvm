! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
SUBROUTINE FOO(F_IN, N)
    INTEGER :: N
    INTEGER :: F_IN(N)

    !$OSS TASK SHARED(F_IN)

        !$OSS TASK SHARED(F_IN)
            F_IN(1) = 1;
        !$OSS END TASK
        !$OSS TASKWAIT

    !$OSS END TASK

    !$OSS TASKWAIT
END SUBROUTINE FOO

PROGRAM P
    IMPLICIT NONE
    INTEGER, PARAMETER :: N = 10
    INTEGER :: F(N)

    CALL FOO(F, N)
END PROGRAM P
