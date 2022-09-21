! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>

PROGRAM P
    IMPLICIT NONE
    INTEGER, PARAMETER :: N = 1000
    INTEGER, PARAMETER :: MAX_TASKS =20
    INTEGER :: I, NTASKS
    LOGICAL :: FIRST_TIME
    INTEGER :: VAR

    FIRST_TIME = .TRUE.
    DO NTASKS=1, MAX_TASKS

        VAR = 0
        !$OSS TASKLOOP GRAINSIZE(NTASKS) FIRSTPRIVATE(FIRST_TIME) SHARED(VAR)
        DO I=1, N, 1
            IF (FIRST_TIME) THEN
                FIRST_TIME = .FALSE.
                !$OSS ATOMIC
                VAR = VAR + 1
            ENDIF
        ENDDO
        !$OSS TASKWAIT

        IF (VAR /= NTASKS) STOP -1


        VAR = 0
        !$OSS TASKLOOP GRAINSIZE(NTASKS) FIRSTPRIVATE(FIRST_TIME) SHARED(VAR)
        DO I=N, 1, -1
            IF (FIRST_TIME) THEN
                FIRST_TIME = .FALSE.
                !$OSS ATOMIC
                VAR = VAR + 1
            ENDIF
        ENDDO
        !$OSS TASKWAIT

        IF (VAR /= NTASKS) STOP -1
    ENDDO
END PROGRAM P
