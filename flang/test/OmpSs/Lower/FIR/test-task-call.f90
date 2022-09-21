! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! XFAIL: true

MODULE MOO
CONTAINS
  SUBROUTINE S1(X)
    IMPLICIT NONE
    INTEGER :: X
    x = x + 1
  END SUBROUTINE S1
END MODULE MOO


PROGRAM MAIN
  USE MOO, ONLY : S1
  IMPLICIT NONE
  INTEGER :: X
  X = 1
  !$OSS TASK
  CALL S1(X)
  !$OSS END TASK
  !$OSS TASKWAIT
END PROGRAM MAIN
