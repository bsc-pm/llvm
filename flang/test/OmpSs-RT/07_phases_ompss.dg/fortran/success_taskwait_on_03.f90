! RUN: %oss-compile-and-run
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_nolink=yes
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER :: X

    !$OSS TASKWAIT ON(X)

    !$OSS TASKWAIT IN(X)
    !$OSS TASKWAIT OUT(X)
    !$OSS TASKWAIT INOUT(X)

    !$OSS TASKWAIT DEPEND(IN: X)
    !$OSS TASKWAIT DEPEND(OUT: X)
    !$OSS TASKWAIT DEPEND(INOUT: X)
END PROGRAM P
