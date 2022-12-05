! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
MODULE MOD_TYPES
    IMPLICIT NONE

    TYPE, PUBLIC :: MY_TYPE
        INTEGER :: X
    END TYPE MY_TYPE
END MODULE MOD_TYPES

MODULE M
    IMPLICIT NONE

    CONTAINS
        !$OSS TASK
    SUBROUTINE FOO(VAR)
        USE MOD_TYPES
        IMPLICIT NONE
        TYPE(MY_TYPE), VALUE :: VAR
        VAR % X = 1
    END SUBROUTINE FOO

    SUBROUTINE FII()
        USE MOD_TYPES
        IMPLICIT NONE
        TYPE(MY_TYPE) :: VAR

    VAR % X = -1
    CALL FOO(VAR)
    !$OSS TASKWAIT
    IF (VAR % X /= -1) STOP 1
    END SUBROUTINE FII
END MODULE M


PROGRAM P
    USE M
    IMPLICIT NONE

    CALL FII()
END PROGRAM P

