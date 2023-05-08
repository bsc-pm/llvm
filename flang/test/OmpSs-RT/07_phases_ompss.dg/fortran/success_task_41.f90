! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
MODULE MOD_TASKS
    IMPLICIT NONE
    CONTAINS
        !$OSS TASK INOUT(MY_ARRAY)
        SUBROUTINE SUB1(N, MY_ARRAY)
            IMPLICIT NONE
            INTEGER, VALUE :: N
            INTEGER :: MY_ARRAY(N)
            MY_ARRAY(:) = MY_ARRAY(:) + 1
        END SUBROUTINE SUB1
END MODULE MOD_TASKS

MODULE MOD_GLOBALS
    IMPLICIT NONE
    INTEGER :: MY_ARRAY(100)
END MODULE MOD_GLOBALS

SUBROUTINE SUB2
    USE MOD_GLOBALS
    USE MOD_TASKS
    IMPLICIT NONE

    MY_ARRAY = 1
    CALL SUB1(100, MY_ARRAY)
    !$OSS TASKWAIT

    IF (ANY(MY_ARRAY /= 2)) STOP 1
END SUBROUTINE SUB2

PROGRAM MAIN
    CALL SUB2
END PROGRAM MAIN

