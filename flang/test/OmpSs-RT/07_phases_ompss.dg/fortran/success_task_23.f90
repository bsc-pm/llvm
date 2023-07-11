! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER, ALLOCATABLE :: V(:, :)

    ALLOCATE(V(10, 5))

    V = 1
    !$OSS TASK INOUT(V)
        V(1, :)= -2
    !$OSS END TASK

    !$OSS TASKWAIT
    if (ANY(V(1,:) /= -2)) STOP 1
    if (ANY(V(2:,:) /= 1)) STOP 2
END PROGRAM P
