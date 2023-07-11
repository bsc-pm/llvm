! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=config/mercurium-ompss
! </testinfo>
MODULE M
    IMPLICIT NONE

    REAL(8), ALLOCATABLE :: V(:)
    INTEGER :: N

    INTERFACE
        !$OSS TASK INOUT(V)
        SUBROUTINE SMP_TASK(V, N)
            IMPLICIT NONE
            INTEGER, VALUE :: N
            REAL(8) :: V(N)
        END SUBROUTINE SMP_TASK
    END INTERFACE
    CONTAINS
        SUBROUTINE S()
            IMPLICIT NONE
            N = 10
            ALLOCATE(V(N))
            V = -1


            CALL SMP_TASK(V, N)
            !$OSS TASKWAIT
            IF (ANY(V /= 3) .and. ANY(V /= 2)) THEN
                PRINT *, V
                STOP -1
            END IF

            ! As we are using 'implements', we don't know what version will be used
            CALL SMP2_TASK(V, N)
            !$OSS TASKWAIT
            IF (ANY(V /= 2)) THEN
                PRINT *, V
                STOP -2
            END IF

        END SUBROUTINE S

        !$OSS TASK INOUT(V)
        SUBROUTINE SMP2_TASK(V, N)
            IMPLICIT NONE
            INTEGER, VALUE :: N
            REAL(8) :: V(N)

            V = 2
        END SUBROUTINE SMP2_TASK
END MODULE M


SUBROUTINE SMP_TASK(V, N)
    IMPLICIT NONE
    INTEGER, VALUE :: N
    REAL(8) :: V(N)

    V = 3
END SUBROUTINE SMP_TASK


PROGRAM P
    USE M
    IMPLICIT NONE

    CALL S()

END PROGRAM P
