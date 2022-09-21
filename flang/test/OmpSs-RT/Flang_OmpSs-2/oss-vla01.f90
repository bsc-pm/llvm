! RUN: %oss-compile-and-run

subroutine task(X)
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY(4: x + 6,2:3)
    INTEGER :: ARRAY2(x + 6:10,2:3)
    INTEGER :: ARRAY3(4:10,x+2:x*5)

    INTEGER :: I,J

    !$OSS TASK SHARED(I)
      I = UBOUND(ARRAY, DIM = 1)
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (I /= x+6) STOP 1

    !$OSS TASK SHARED(I)
      I = LBOUND(ARRAY2, DIM = 1)
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (I /= x+6) STOP 1

    !$OSS TASK SHARED(I, J)
      I = LBOUND(ARRAY3, DIM = 2)
      J = UBOUND(ARRAY3, DIM = 2)
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (I /= x+2) STOP 1
    IF (J /= x*5) STOP 1

    !$OSS TASK SHARED(I, ARRAY)
      I = UBOUND(ARRAY, DIM = 1)
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (I /= x+6) STOP 1

    !$OSS TASK SHARED(I, ARRAY)
      I = LBOUND(ARRAY2, DIM = 1)
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (I /= x+6) STOP 1

    !$OSS TASK SHARED(I, J, ARRAY)
      I = LBOUND(ARRAY3, DIM = 2)
      J = UBOUND(ARRAY3, DIM = 2)
    !$OSS END TASK
    !$OSS TASKWAIT

    IF (I /= x+2) STOP 1
    IF (J /= x*5) STOP 1
end subroutine

PROGRAM MAIN
    IMPLICIT NONE
    INTEGER :: A

    A = 3

    CALL task(A)

END PROGRAM MAIN
