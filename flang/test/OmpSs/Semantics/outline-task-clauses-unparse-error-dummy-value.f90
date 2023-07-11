! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

MODULE MOO
CONTAINS
  !ERROR: In a task outline 'y' cannot be a Value
  !$OSS TASK INOUT(Y)
  SUBROUTINE S2(X,Y)
    IMPLICIT NONE
    INTEGER :: X
    INTEGER,VALUE :: Y

    Y = 1
    X = Y + 1
  END SUBROUTINE S2
END MODULE MOO

PROGRAM MAIN
  USE MOO, ONLY : S2
  IMPLICIT NONE
  INTEGER :: X
  INTEGER :: Y

  INTERFACE
    !ERROR: In a task outline 'y' cannot be a Value
    !$OSS TASK INOUT(X, Y)
    SUBROUTINE S1(X,Y)
      IMPLICIT NONE
      INTEGER :: X
      INTEGER, VALUE :: Y
    END SUBROUTINE S1
  END INTERFACE

  X = 1
  Y = 0
  CALL S1(X,Y)
  CALL S2(X,Y)
  !$OSS TASKWAIT
  IF (X /= 3) STOP "ERROR"
END PROGRAM MAIN 

