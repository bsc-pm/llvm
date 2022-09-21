! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=config/mercurium-nanos6-regions
! </testinfo>

!$OSS TASK OUT(X)
SUBROUTINE S1(X)
  IMPLICIT NONE
  INTEGER :: X
  X = X + 1
  PRINT *, "Voy Primero"
END SUBROUTINE S1

MODULE MOO
CONTAINS
  !$OSS TASK IN(X2)
  SUBROUTINE S2(X2)
    IMPLICIT NONE
    INTEGER :: X2
    X2 = X2 + 2
    PRINT *, "Voy Segundo"
  END SUBROUTINE S2
END MODULE MOO

PROGRAM MAIN
  USE MOO, ONLY : S2
  IMPLICIT NONE
  INTEGER :: Y
  Y = 1
  CALL S1(Y)
  CALL S2(Y)
  !$OSS TASKWAIT
END PROGRAM MAIN 
