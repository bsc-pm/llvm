! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=config/mercurium-ompss
! </testinfo>
FUNCTION OMP_IN_FINAL() RESULT(X)
    IMPLICIT NONE
    LOGICAL X
    STOP -1
    X = .FALSE.
END FUNCTION OMP_IN_FINAL

RECURSIVE FUNCTION F(N) RESULT(RES)
    IMPLICIT NONE
    INTEGER :: N, RES
    LOGICAL :: X
    LOGICAL, EXTERNAL  :: OMP_IN_FINAL
    RES = 0
    IF (N > 0) THEN
    !$OSS TASK SHARED(RES)
        RES = F(N-1)
        IF(OMP_IN_FINAL()) THEN
            RES = RES + 1
        ELSE
            RES = RES + 2
        ENDIF
    !$OSS END TASK
    ENDIF
END FUNCTION F

PROGRAM P
    IMPLICIT NONE
    INTEGER :: RES
    INTEGER, EXTERNAL :: F
    RES = 0

    !$OSS TASK SHARED(RES) FINAL(.TRUE.)
        RES = F(10)
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (RES /= 10) STOP -2
END PROGRAM P
