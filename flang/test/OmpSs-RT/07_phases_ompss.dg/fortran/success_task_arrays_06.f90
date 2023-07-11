! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    INTERFACE
        !$OSS TASK
        SUBROUTINE FOO(S, L1, L2)
            IMPLICIT NONE
            INTEGER :: L1, L2
            INTEGER :: S(1:L1, 1:L2, 1:19)
        END SUBROUTINE FOO
    END INTERFACE

    INTEGER :: A(1:200, 1:100, 1:19)

    A = 0

    CALL FOO(A, 100, 200)

    !$OSS TASKWAIT

    IF (ANY(A /= 1)) STOP 1
END PROGRAM P

SUBROUTINE FOO(S, L1, L2)
    IMPLICIT NONE
    INTEGER :: L1, L2
    INTEGER :: S(1:L1, 1:L2, 1:19)

    S(:, :, :) = 1
END SUBROUTINE FOO
