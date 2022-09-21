! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER, ALLOCATABLE :: X(:)

    ALLOCATE(X(1000))

    X = 0
    !$OSS TASK FIRSTPRIVATE(X)
        X = 1
    !$OSS END TASK
    !$OSS TASKWAIT
    IF (ANY(X /= 0)) STOP 1
END PROGRAM P
