! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_compile_fail=yes
! </testinfo>
SUBROUTINE F(V, LEN)
    IMPLICIT NONE
    INTEGER ::V(*)
    iNTEGER :: LEN

    !$OSS TASK FIRSTPRIVATE(V)
        PRINT *, V(1)
    !$OSS END TASK
END SUBROUTINE F
