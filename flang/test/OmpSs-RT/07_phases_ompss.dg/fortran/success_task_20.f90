! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
SUBROUTINE FOO(X)
    INTEGER :: X
    X = X + 1
END SUBROUTINE FOO

PROGRAM P
    IMPLICIT NONE

    INTERFACE
        !$OSS TASK INOUT(X) FINAL(.TRUE.)
        SUBROUTINE FOO(X)
            INTEGER :: X
        END SUBROUTINE FOO
    END INTERFACE
    INTEGER X

    X = -1

     CALL FOO(X)
!$OSS TASKWAIT ON(X)
IF (X /= 0) THEN
    STOP -1
END IF

    !$OSS TASK INOUT(X) FINAL(.TRUE.)
         X = X + 1
     !$OSS END TASK
!$OSS TASKWAIT ON(X)

IF (X /= 1) THEN
    STOP -1
END IF

END PROGRAM P
