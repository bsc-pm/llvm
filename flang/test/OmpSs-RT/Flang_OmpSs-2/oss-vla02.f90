! RUN: %oss-compile-and-run

subroutine task(X)
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY(x+2:x*5)

    INTEGER :: I

    DO I=X+2,X*5
      ARRAY(I) = I
    END DO

    !$OSS TASK FIRSTPRIVATE(ARRAY)
      DO I=X+2,X*5
        ARRAY(I) = I+1
      END DO
    !$OSS END TASK
    !$OSS TASKWAIT

    DO I=X+2,X*5
      IF (ARRAY(I) /= I) STOP 1
    END DO

    !$OSS TASK SHARED(ARRAY)
      DO I=X+2,X*5
        ARRAY(I) = I+2
      END DO
    !$OSS END TASK
    !$OSS TASKWAIT

    DO I=X+2,X*5
      IF (ARRAY(I) /= I+2) STOP 1
    END DO
end subroutine

subroutine task2(X,Y)
    IMPLICIT NONE

    INTEGER :: X,Y
    INTEGER :: ARRAY(x+2:x*5,y+3:y*6)

    INTEGER :: I,J

    DO J=Y+3,Y*6
      DO I=X+2,X*5
        ARRAY(I,J) = I+J
      END DO
    END DO

    !$OSS TASK FIRSTPRIVATE(ARRAY)
      DO J=Y+3,Y*6
        DO I=X+2,X*5
          ARRAY(I,J) = (I+J)*2
        END DO
      END DO
    !$OSS END TASK
    !$OSS TASKWAIT

    DO J=Y+3,Y*6
      DO I=X+2,X*5
        IF (ARRAY(I,J) /= I+J) STOP 1
      END DO
    END DO

    !$OSS TASK SHARED(ARRAY)
      DO J=Y+3,Y*6
        DO I=X+2,X*5
          ARRAY(I,J) = (I+J)*3
        END DO
      END DO
    !$OSS END TASK
    !$OSS TASKWAIT

    DO J=Y+3,Y*6
      DO I=X+2,X*5
        IF (ARRAY(I,J) /= (I+J)*3) STOP 1
      END DO
    END DO
end subroutine

PROGRAM MAIN
    IMPLICIT NONE
    INTEGER :: A,B

    A = 3
    B = 4

    CALL task(A)
    CALL task2(A,B)

END PROGRAM MAIN
