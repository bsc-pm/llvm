! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM TEST
    IMPLICIT NONE

    TYPE T
        INTEGER :: Q(4)
    END TYPE T

    TYPE(T), ALLOCATABLE :: V(:)

    INTEGER :: INDEX1, UBOUND1
    REAL*8 :: RESULTS


    ALLOCATE(V(2))

    INDEX1 = 1
    UBOUND1= 2

    !$OSS TASK FIRSTPRIVATE(INDEX1, UBOUND1) INOUT(V(INDEX1) % Q(1:UBOUND1))
    !$OSS END TASK

    !$OSS TASKWAIT

    DEALLOCATE(V)
END PROGRAM TEST

