! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    INTEGER I,J

    I = 0
    J = 1
    !$OSS TASK INOUT(i,j)
        if (i == 0) J = 0

        !$OSS TASK INOUT(i, j)
        I = I + 1
        !$OSS END TASK
        !$OSS TASKWAIT

       J = J + 1
    !$OSS END TASK
    !$OSS TASKWAIT

    if (I /= 1 .or. J /= 1) STOP 1
END PROGRAM P
