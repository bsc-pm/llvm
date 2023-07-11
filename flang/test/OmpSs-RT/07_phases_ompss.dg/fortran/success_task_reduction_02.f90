! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_compile_fail_nanos6_mercurium=yes
! test_compile_fail_nanos6_imfc=yes
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER, PARAMETER :: N = 100
    INTEGER, ALLOCATABLE :: X(:), Y(:, :)

    ALLOCATE(X(N), Y(N, N))

    !$OSS TASK REDUCTION(+: X) REDUCTION(+: Y)
        X = 0
        Y = 1
    !$OSS END TASK
    !$OSS TASKWAIT
END PROGRAM P
