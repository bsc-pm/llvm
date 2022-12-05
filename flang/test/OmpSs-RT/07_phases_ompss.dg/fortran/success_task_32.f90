! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM MAIN
    IMPLICIT NONE
    INTERFACE
        !$OSS TASK INOUT(X, Y)
        SUBROUTINE S(X, Y)
            INTEGER :: X, Y
            OPTIONAL :: Y
        END SUBROUTINE S
    END INTERFACE

    INTEGER :: X, Y

    X = 1
    Y = 2

    CALL S(X)
    !$OSS TASKWAIT
    IF (X /= 2) STOP 1
    IF (Y /= 2) STOP 2


    CALL S(X, Y)
    !$OSS TASKWAIT
    IF (X /= 3) STOP 1
    IF (Y /= 3) STOP 2

END PROGRAM MAIN

SUBROUTINE S(X, Y)
   IMPLICIT NONE
   INTEGER :: X, Y
   OPTIONAL :: Y

   X = X + 1
   IF (PRESENT(Y)) Y = Y + 1
END SUBROUTINE S
