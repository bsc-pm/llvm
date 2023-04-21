! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
MODULE TEST1
    IMPLICIT NONE
    INTEGER :: IPROVA

CONTAINS

    SUBROUTINE PROBANDO
        IPROVA=IPROVA+50
    END SUBROUTINE PROBANDO

END MODULE

PROGRAM MAIN
    USE TEST1
    IMPLICIT NONE

    IPROVA = 0
    !$OSS TASK FIRSTPRIVATE(IPROVA)
    IPROVA = 5
    CALL PROBANDO()
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (IPROVA /= 50) THEN
        PRINT *, IPROVA
        STOP 1
    END IF

END PROGRAM MAIN
