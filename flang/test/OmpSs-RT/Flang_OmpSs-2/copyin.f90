! RUN: %oss-compile-and-run | FileCheck %s

PROGRAM P
REAL, ALLOCATABLE :: X(:, :)

ALLOCATE(X(5, 5))
X = 888
X(4, 1) = 1
X(4, 2) = 2
X(4, 3) = 3
X(4, 4) = 4
X(4, 5) = 5
CALL test_intent_in(x(4, :))

contains
subroutine test_intent_in(x)
  real :: x(:)
  call bar_intent_in(x)
!$OSS TASKWAIT
end subroutine

!$OSS TASK
subroutine bar_intent_in(x)
  real, intent(in) :: x(5)
PRINT *, x
end subroutine
end

!CHECK: 1. 2. 3. 4. 5.
