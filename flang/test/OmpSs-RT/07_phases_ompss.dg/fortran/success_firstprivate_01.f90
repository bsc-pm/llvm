! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER :: X(10)

    X = 0
    !$OSS TASK FIRSTPRIVATE(X)
        X = 1
    !$OSS END TASK
    !$OSS TASKWAIT

    PRINT *, X

    IF (ANY(X /= 0)) STOP 1
END PROGRAM P
