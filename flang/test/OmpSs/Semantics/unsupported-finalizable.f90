! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

module P
  TYPE TY
  CONTAINS
  FINAL :: foo
  END TYPE
CONTAINS
SUBROUTINE foo(t)
 type(ty) :: t
END
END

PROGRAM PROG 
  use P
  implicit none
  type(ty) :: t
  !ERROR: finalizable type is not supported
  !$OSS TASK FIRSTPRIVATE(t)
  !$OSS END TASK
END
