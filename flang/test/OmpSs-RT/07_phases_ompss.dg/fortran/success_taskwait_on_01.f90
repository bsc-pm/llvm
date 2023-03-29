! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER :: X(10), Y(10)
    INTEGER :: I

    X = 1
    Y = 2

    !$OSS TASK INOUT(X(2:3), Y(2:3)) FIRSTPRIVATE(I)
       DO I = 2, 3
          X(I) = X(I) + 1
          Y(I) = Y(I) + 1
       END DO
       PRINT  *, "TASK DONE"
    !$OSS END TASK


    !$OSS TASKWAIT ON(X(2:3), Y(2:3))

    DO I = 2, 3
       IF (X(I) /= 2) STOP "ERROR 1"
       IF (Y(I) /= 3) STOP "ERROR 2"
    END DO

    PRINT *, "OK!"
END PROGRAM P
