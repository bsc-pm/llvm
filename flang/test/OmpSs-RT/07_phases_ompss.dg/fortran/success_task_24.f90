! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER, POINTER :: PTR(:, :)

    ALLOCATE(PTR(10, 5))

    PTR = 1

    !$OSS TASK INOUT(PTR)
        PTR(1, :)= -2
    !$OSS END TASK

    !$OSS TASKWAIT
    if (ANY(PTR(1,:) /= -2)) STOP 1
    if (ANY(PTR(2:,:) /= 1)) STOP 2
END PROGRAM P
