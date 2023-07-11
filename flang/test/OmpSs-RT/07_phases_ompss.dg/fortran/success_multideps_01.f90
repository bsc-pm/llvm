! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_FFLAGS="--no-copy-deps"
! test_ENV="NX_THROTTLE=dummy"
! </testinfo>
MODULE MOO
    INTEGER, PARAMETER :: NUM_ITEMS = 100
    INTEGER, PARAMETER :: NUM_TASKS = 10000
    INTEGER VEC(NUM_ITEMS)

    CONTAINS

        SUBROUTINE GENERATE(LENGTH, A, VAL)
            IMPLICIT NONE
            INTEGER :: LENGTH
            INTEGER :: A(LENGTH)
            INTEGER :: VAL

            INTEGER :: I

            !$OSS TASK PRIVATE(I) FIRSTPRIVATE(LENGTH) OUT( [A(K), K = 1, LENGTH] )
            DO I = 1, LENGTH
                A(I) = VAL
            END DO
            !$OSS END TASK

        END SUBROUTINE GENERATE

        SUBROUTINE CONSUME(LENGTH, A, VAL)
            IMPLICIT NONE
            INTEGER :: LENGTH
            INTEGER :: A(LENGTH)
            INTEGER :: VAL

            INTEGER :: I

            !$OSS TASK PRIVATE(I) FIRSTPRIVATE(LENGTH) IN( [A(K), K = 1, LENGTH] )
            DO I = 1, LENGTH
                IF (A(I) /= VAL) STOP 1
            END DO
            !$OSS END TASK

        END SUBROUTINE CONSUME

END MODULE

PROGRAM MAIN
    USE MOO
    IMPLICIT NONE

    INTEGER :: I

    INTEGER :: START, LENGTH, VAL

    PRINT *, "INITIALIZING", NUM_ITEMS, "ITEMS"

    DO I = 1, NUM_ITEMS
        VEC(I) = I
    END DO

    PRINT *, "CREATING", NUM_TASKS, "TASKS"

    DO I = 1, NUM_TASKS
        START = MOD(I, NUM_ITEMS)
        LENGTH = 20

        IF ( START + LENGTH >= NUM_ITEMS ) LENGTH = NUM_ITEMS - START

        VAL = I

        CALL GENERATE(LENGTH, VEC(START), VAL)
        CALL CONSUME(LENGTH, VEC(START), VAL)
    END DO

    PRINT *, "WAITING TASKS"
    !$OSS TASKWAIT
END PROGRAM MAIN
