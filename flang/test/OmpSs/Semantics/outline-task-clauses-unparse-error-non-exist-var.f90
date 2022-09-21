! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

!ERROR: In a task outline 'z' is undefined
!$OSS TASK INOUT(Z)
SUBROUTINE S1(X)
  IMPLICIT NONE
  INTEGER :: X

  X = X + 1
END SUBROUTINE S1

! TODO: MODULE

MODULE MOO
INTEGER :: Z
CONTAINS
  !ERROR: In a task outline 'z' must be a dummyArgument
  !$OSS TASK INOUT(Z)
  SUBROUTINE S2(X)
    IMPLICIT NONE
    INTEGER :: X

    X = X + 1
  END SUBROUTINE S2
END MODULE MOO

PROGRAM MAIN
  USE MOO, ONLY : S2
  IMPLICIT NONE
  INTEGER :: X
  X = 1
  CALL S1(X)
  CALL S2(X)
  !$OSS TASKWAIT
  IF (X /= 3) STOP "ERROR"
END PROGRAM MAIN 

