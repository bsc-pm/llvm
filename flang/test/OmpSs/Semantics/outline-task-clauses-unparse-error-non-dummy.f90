! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

!ERROR: In a task outline 'y' must be a dummyArgument
!$OSS TASK INOUT(Y)
SUBROUTINE S1(X)
  IMPLICIT NONE
  INTEGER :: X
  INTEGER :: Y

  Y = X
  X = Y + 1
END SUBROUTINE S1

MODULE MOO
CONTAINS
  !ERROR: In a task outline 'y' must be a dummyArgument
  !$OSS TASK INOUT(Y)
  SUBROUTINE S2(X)
    IMPLICIT NONE
    INTEGER :: X
    INTEGER :: Y

    Y = X
    X = Y + 1
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

