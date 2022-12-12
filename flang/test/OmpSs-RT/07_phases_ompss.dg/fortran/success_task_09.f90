! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM MAIN
  IMPLICIT NONE

  INTEGER, ALLOCATABLE :: X(:), Y(:)

  ALLOCATE(X(10), Y(10))

  !$OSS TASK OUT(X)
  X = 1
  !$OSS END TASK

  !$OSS TASK IN(X) OUT(Y)
  Y = X + 3
  !$OSS END TASK

  !$OSS TASKWAIT

  IF (ANY(Y(:) /= 4)) STOP 1
END PROGRAM MAIN

