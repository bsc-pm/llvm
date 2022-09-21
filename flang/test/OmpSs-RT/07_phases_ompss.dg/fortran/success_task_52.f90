! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>

PROGRAM P
    IMPLICIT NONE
    INTEGER :: I, N, X
    LOGICAL :: FLAG

    X = 0
    N = 1
    FLAG = .FALSE.


    !$OSS TASK IF (.NOT. FLAG .AND. N > 0 ) SHARED(X)
    !$OSS ATOMIC
     X = X + 1
    !$OSS END TASK
    !$OSS TASKWAIT
    IF (X /= 1) STOP 1
END PROGRAM P
