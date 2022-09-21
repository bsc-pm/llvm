! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_compile_fail_nanos6_mercurium=yes
! test_compile_fail_nanos6_imfc=yes
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

    !$OSS DECLARE REDUCTION(foo: INTEGER : OMP_OUT = OMP_OUT + OMP_IN)

    RES = 0
    DO I=1, N
        !$OSS TASK REDUCTION(foo: RES) SHARED(V) FIRSTPRIVATE(I)
            RES = RES + V(I)
        !$OSS END TASK
    ENDDO
   !$OSS TASKWAIT

   IF (RES /= ((N*(N+1))/2)) STOP 1

END PROGRAM P
