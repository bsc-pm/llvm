! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_compile_fail_nanos6_mercurium=yes
! test_compile_fail_nanos6_imfc=yes
! </testinfo>
PROGRAM MAIN
    IMPLICIT NONE

    INTEGER :: X, Z, ORIG_X

    X = 1
    ORIG_X = 1
    Z = 2
    CALL INNER(X)

    !$OSS TASKWAIT

    IF (X /= (ORIG_X + Z)) THEN
        PRINT "('X SHOULD BE ',I0,' BUT IS ', I0)", ORIG_X + Z , X
        STOP 1
    END IF

    CONTAINS
        !$OSS TASK INOUT(Y, Z)
        SUBROUTINE INNER(Y)
            IMPLICIT NONE
            INTEGER :: Y
            Y = Y + Z
        END SUBROUTINE INNER
END PROGRAM MAIN
