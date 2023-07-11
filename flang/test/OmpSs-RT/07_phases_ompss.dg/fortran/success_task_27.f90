! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>

PROGRAM P
    IMPLICIT NONE
    INTEGER :: I, LIMIT
        !$OSS TASK FINAL(.FALSE.)
        !$OSS END TASK

        !$OSS TASK FINAL(I < LIMIT)
        !$OSS END TASK

        !$OSS TASKWAIT
END PROGRAM P
