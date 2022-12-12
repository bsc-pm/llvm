! RUN: %oss-compile-and-run

! check to see we do not care about the stride (local vars)
PROGRAM P
  CALL FOO()
CONTAINS
SUBROUTINE FOO()
  IMPLICIT NONE
  INTEGER, ALLOCATABLE :: A(:, :)

  ALLOCATE(A(10, 20))
  A = 777

  !$OSS TASK OUT(A(2:5, 2:10))
  A(2, 2) = 4
  !$OSS END TASK
  !$OSS TASK IN(A(2:5:2, 2:10:2))
  IF (A(2, 2) /= 4) STOP 1 
  !$OSS END TASK
  !$OSS TASKWAIT
END SUBROUTINE
END PROGRAM
