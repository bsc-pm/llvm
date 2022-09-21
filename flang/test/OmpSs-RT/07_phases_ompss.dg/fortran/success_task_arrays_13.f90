! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER :: X(10, 20) 

    X = 1
    CALL FOO(X)
    !$OSS TASKWAIT

    IF (ANY(X(1, 1:10) /= 2)) STOP 1
END PROGRAM P

SUBROUTINE FOO(X)
IMPLICIT NONE
INTEGER :: X(10, *)

!$OSS TASK INOUT(X(1, 1:10))
   X(1, 1:10) = X(1, 1:10) + 1
!$OSS END TASK

END SUBROUTINE FOO
