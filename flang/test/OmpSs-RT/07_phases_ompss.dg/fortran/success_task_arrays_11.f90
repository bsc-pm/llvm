! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM MAIN
    IMPLICIT NONE

    INTEGER, DIMENSION(:), ALLOCATABLE :: IPROVA
    ALLOCATE(IPROVA(5))

    !$OSS TASK inout(IPROVA(1:5))
        IPROVA(1)=5
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (IPROVA(1) /= 5) STOP 1

END PROGRAM
