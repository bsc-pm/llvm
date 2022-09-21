! RUN: %oss-compile-and-run

MODULE FOO
    CONTAINS
        SUBROUTINE BAR(X)
            INTEGER :: X

            X = X + 1
        END SUBROUTINE BAR
END MODULE FOO

PROGRAM MAIN
    USE FOO
    IMPLICIT NONE
    INTEGER :: Y

    Y = 3

    CALL BAR(Y)
    IF (Y /= 4) STOP 1

END PROGRAM MAIN

