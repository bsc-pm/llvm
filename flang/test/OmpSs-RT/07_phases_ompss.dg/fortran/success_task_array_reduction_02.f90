! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
SUBROUTINE FOO(V, RES, N, M)
    IMPLICIT NONE
    INTEGER :: N, M
    INTEGER :: V(N, M)
    INTEGER :: RES(10)
    INTEGER :: I

    RES = 0
    DO I=1, M
        !$OSS TASK REDUCTION(+: RES) SHARED(V)
            !! PRINT *, "------> ", RES
            RES = RES + V(:, I)
        !$OSS END TASK
    ENDDO
    !$OSS TASKWAIT

END SUBROUTINE FOO


PROGRAM P
    IMPLICIT NONE

    INTEGER, PARAMETER :: N = 10
    INTEGER, PARAMETER :: M = 10
    INTEGER :: V(N, M)
    INTEGER :: I,J
    INTEGER :: RES(N)

    DO J=1,M
        DO I=1,N
            V(I, J) =  ((J-1)*M) + I
        ENDDO
    ENDDO

    CALL FOO(V, RES, N, M)

    !! PRINT *, SUM(RES), ((N*M*(N*M+1))/2)
    IF (SUM(RES) /= ((N*M*(N*M+1))/2))  STOP 1

    PRINT *, "ok!"
END PROGRAM P
