! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    INTEGER :: X(20, 10)

    X = 1

    !$OSS TASK INOUT(X(2, 1))
       X(2, 1) = X(2, 1) + 41
    !$OSS END TASK


    !$OSS TASK IN(X(2, 1))
        IF (X(2,1) /= 42) STOP 1
    !$OSS END TASK

    !$OSS TASKWAIT
END PROGRAM P
