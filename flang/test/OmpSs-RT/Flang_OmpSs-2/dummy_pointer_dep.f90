! RUN: %oss-compile-and-run | FileCheck %s

! check to see we register properly a dep chain using pointers
! to the same buffer (dummy)
PROGRAM P
  IMPLICIT NONE
  INTEGER, TARGET :: TMP(4, 4)
  INTEGER, POINTER :: PTR0(:, :)
  INTEGER, POINTER :: PTR1(:, :)

  TMP = 777

  PTR0 => TMP
  PTR1 => TMP(2:4, 2:4)

  CALL FOO(PTR0, PTR1)
CONTAINS
SUBROUTINE FOO(PTR0, PTR1)
  IMPLICIT NONE
  INTEGER, POINTER :: PTR0(:, :)
  INTEGER, POINTER :: PTR1(:, :)

  !$OSS TASK OUT(PTR0(2, 2))
  PTR0(2, 2) = 4
  PRINT *, PTR0
  !$OSS END TASK

  !$OSS TASK IN(PTR1)
  PRINT *, PTR1
  !$OSS END TASK

  !$OSS TASKWAIT
END SUBROUTINE
END PROGRAM

! CHECK: 777 777 777 777 777 4 777 777 777 777 777 777 777 777 777 777
! CHECK: 4 777 777 777 777 777 777 777 777
