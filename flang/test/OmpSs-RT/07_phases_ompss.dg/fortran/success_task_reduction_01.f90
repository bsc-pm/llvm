! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER, PARAMETER :: N = 1000
    INTEGER :: V(N)
    INTEGER :: I
    INTEGER :: RES

    DO I=1, N
        V(I) = I
    ENDDO

    RES = 0
    DO I=1, N
        !$OSS TASK REDUCTION(+: RES) SHARED(V) FIRSTPRIVATE(I)
            RES = RES + V(I)
        !$OSS END TASK
    ENDDO

    !$OSS TASK IN(RES)
        IF (RES /= ((N*(N+1))/2)) STOP 1
    !$OSS END TASK

    !$OSS TASKWAIT

END PROGRAM P
