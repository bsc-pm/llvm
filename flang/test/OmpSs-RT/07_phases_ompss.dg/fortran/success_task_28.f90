! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_nolink=yes
! </testinfo>

SUBROUTINE FOO(N)
    IMPLICIT NONE
    INTEGER :: N
    INTEGER :: A(N)
    INTEGER :: B(N)

    !$OSS TASK SHARED(A) INOUT(B)
    !$OSS END TASK

    !$OSS TASKWAIT
END SUBROUTINE FOO
