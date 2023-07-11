! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_FFLAGS="--no-copy-deps"
! </testinfo>

SUBROUTINE FOO()
    IMPLICIT NONE
    INTEGER :: X, Y

    X = 0
    Y = 2

    !$OSS TASK DEPEND(INOUT: X) SHARED(X)
        X = X + 1
    !$OSS END TASK

    !$OSS TASK DEPEND(INOUT: Y) SHARED(Y)
        Y = Y - 1
    !$OSS END TASK

    !$OSS TASKWAIT DEPEND(IN: X)
    IF (X /= 1) STOP -1

    !!! POTENTIALLY RACE CONDITION, NOTE THAT THE SECOND TASK MAY NOT BE
    !!! EXECUTED AT THIS POINT!
    !!! IF(X /= Y) STOP -2

    !$OSS TASKWAIT
    IF (X /= Y) STOP -2
END SUBROUTINE FOO

PROGRAM P
    IMPLICIT NONE
    CALL FOO()
END PROGRAM P
