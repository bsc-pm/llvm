! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
MODULE M
    CONTAINS
        SUBROUTINE S2(A)
            IMPLICIT NONE
            INTEGER :: A(:)

            CALL ST(A, 2)
        END SUBROUTINE S2

        SUBROUTINE S3(A)
            IMPLICIT NONE
            INTEGER :: A(:)

            CALL ST(A, 3)
        END SUBROUTINE S3

        !$OSS TASK INOUT(X(1:10))
        SUBROUTINE ST(X, Z)
            INTEGER :: X(:), Z

            X(1:10) = Z
        END SUBROUTINE ST
END MODULE M

PROGRAM MAIN
    USE M
    IMPLICIT NONE

    INTEGER :: A(100)

    CALL S2(A)
    !$OSS TASKWAIT
    IF (ANY(A(1:10) /= 2)) STOP 1

    CALL S3(A)
    !$OSS TASKWAIT
    IF (ANY(A(1:10) /= 3)) STOP 1
END PROGRAM MAIN
