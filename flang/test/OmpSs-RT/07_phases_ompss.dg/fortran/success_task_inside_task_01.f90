! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM MAIN
    IMPLICIT NONE

    INTERFACE
        SUBROUTINE S(X, Y)
            IMPLICIT NONE
            INTEGER :: X(:, :)
            INTEGER, ALLOCATABLE :: Y(:, :)
        END SUBROUTINE S
    END INTERFACE

    INTEGER :: X(10, 20)
    INTEGER, ALLOCATABLE :: Y(:, :)

    X = 1

    ALLOCATE(Y(10, 20))
    Y = 5

    !$OSS TASK SHARED(X, Y)
        !$OSS TASK INOUT(X, Y)
        X = X + 1
        Y = Y + 1
        !$OSS END TASK
        !$OSS TASKWAIT
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (ANY(X /= 2)) THEN
        STOP 1
    END IF
    IF (ANY(Y /= 6)) THEN
        STOP 2
    END IF

    CALL S(X, Y)

    IF (ANY(X /= 3)) THEN
        STOP 3
    END IF
    IF (ANY(Y /= 7)) THEN
        STOP 4
    END IF

    DEALLOCATE(Y)
END PROGRAM MAIN

SUBROUTINE S(X, Y)
    IMPLICIT NONE
    INTEGER :: X(:, :)
    INTEGER, ALLOCATABLE :: Y(:, :)

    !$OSS TASK SHARED(X) INOUT(X, Y)
        !$OSS TASK INOUT(X, Y)
        X = X + 1
        Y = Y + 1
        !$OSS END TASK
        !$OSS TASKWAIT
    !$OSS END TASK

    !$OSS TASKWAIT
END SUBROUTINE S
