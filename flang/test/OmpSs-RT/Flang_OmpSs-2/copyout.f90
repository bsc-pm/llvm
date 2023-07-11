! RUN: %oss-compile-and-run | FileCheck %s

PROGRAM P
REAL, ALLOCATABLE :: X(:, :)

ALLOCATE(X(5, 5))
X = 888
CALL test_intent_out(x(4, :))
PRINT *, X

contains
subroutine test_intent_out(x)
  real :: x(:)
  call bar_intent_out(x)
!$OSS TASKWAIT
end subroutine

!$OSS TASK
subroutine bar_intent_out(x)
  real, intent(out) :: x(5)
  X(1) = 1
  X(2) = 2
  X(3) = 3
  X(4) = 4
  X(5) = 5
end subroutine
end

!CHECK: 888. 888. 888. 1. 888. 888. 888. 888. 2. 888. 888. 888. 888. 3. 888. 888. 888.
!CHECK: 888. 4. 888. 888. 888. 888. 5. 888.
