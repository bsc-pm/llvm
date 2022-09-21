! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>

PROGRAM MAIN
    IMPLICIT NONE
    INTEGER, TARGET :: T
    INTEGER, POINTER :: P
    !$OSS TASK
    P => T
    !$OSS END TASK

    !$OSS TASKWAIT
END PROGRAM MAIN
