! RUN: %oss-compile-and-run

! check to see we do not care about the stride
! allocatables in derived types
PROGRAM P
  IMPLICIT NONE
  TYPE :: TY
    INTEGER, ALLOCATABLE :: A(:, :)
  END TYPE
  TYPE(TY) :: T

  ALLOCATE(T%A(10, 20))

  T%A = 777

  !$OSS TASK OUT(T%A(2:5, 2:10))
  T%A(2, 2) = 4
  !$OSS END TASK
  !$OSS TASK IN(T%A(2:5:2, 2:10:2))
  IF (T%A(2, 2) /= 4) STOP 1
  !$OSS END TASK
  !$OSS TASKWAIT
END PROGRAM
