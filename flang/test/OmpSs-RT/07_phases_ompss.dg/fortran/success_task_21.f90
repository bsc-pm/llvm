! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    IMPLICIT NONE

    TYPE T0
        INTEGER :: X(10)
    END TYPE T0

    TYPE T2
        INTEGER :: X(10)
    END TYPE T2

    TYPE T1
        TYPE(T2) :: Y
    END TYPE T1

    TYPE(T0) :: VAR1
    TYPE(T1) :: VAR2

    VAR1 % X = -1
    VAR2 % Y % X = -1

    !$OSS TASK INOUT(VAR1 % X)
    VAR1 % X(1) = 1
    !$OSS END TASK
    !$OSS TASKWAIT
    
    IF (VAR1 % X(1) /= 1) STOP 1

    !$OSS TASK INOUT(VAR2 % Y % X)
    VAR2 % Y % X(1) = 1
    !$OSS END TASK
    !$OSS TASKWAIT
    IF (VAR2% Y % X(1) /= 1) STOP 1

END PROGRAM P

