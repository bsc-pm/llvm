! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=config/mercurium-ompss-2
! test_exec_fail=yes
! </testinfo>

SUBROUTINE BAR(X)
IMPLICIT NONE
INTEGER, INTENT(INOUT) :: X
IF (X /= 5) STOP 2
X = 77
END SUBROUTINE

PROGRAM P
IMPLICIT NONE
INTEGER :: X

X = 5

!$OSS TASK ONREADY(BAR(X)) SHARED(X)
!$OSS END TASK
!$OSS TASKWAIT

IF (X /= 77) STOP 1

END PROGRAM
