! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
SUBROUTINE S2(X)
    IMPLICIT NONE
    INTEGER :: X(:, :)

    X = X + 1
END SUBROUTINE S2

SUBROUTINE S(A)
    IMPLICIT NONE
    INTEGER :: A(:, :)
    INTERFACE
        !$OSS TASK INOUT(X)
        SUBROUTINE S2(X)
            IMPLICIT NONE
            INTEGER :: X(:, :)
        END SUBROUTINE S2
    END INTERFACE

    CALL S2(A)
END SUBROUTINE S

PROGRAM MAIN
    IMPLICIT NONE
    INTEGER :: C(5, 4)

    INTERFACE
        SUBROUTINE S(A)
            IMPLICIT NONE
            INTEGER :: A(:, :)
        END SUBROUTINE S
    END INTERFACE

    C = 100

    PRINT *, C

    CALL S(C)
    !$OSS TASKWAIT

    PRINT *, C

    IF (ANY(C /= 101)) STOP 1

END PROGRAM MAIN
