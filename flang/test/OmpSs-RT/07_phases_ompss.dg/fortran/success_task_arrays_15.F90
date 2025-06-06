! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_compile_fail_nanos6_mercurium=yes
! test_compile_fail_nanos6_imfc=yes
! </testinfo>
MODULE MOO
    INTEGER :: Z
END MODULE MOO

PROGRAM P
    USE MOO, ONLY : Z
    IMPLICIT NONE
    TYPE  MY_TYPE
        INTEGER, POINTER :: ARR(:)
    END TYPE  MY_TYPE

    INTERFACE
        !$OSS TASK INOUT(ARR)
        SUBROUTINE FOO(ARR)
            USE MOO, ONLY : Z
            IMPLICIT NONE
            INTEGER, POINTER :: ARR(:)
        END SUBROUTINE FOO
    END INTERFACE

#ifdef __INTEL_COMPILER
    EXTERNAL :: SLEEP
#endif

    TYPE(MY_TYPE) :: V
    Z = 0

    ALLOCATE(V % ARR(100))
    !$OSS TASK INOUT(V % ARR(:))
        PRINT *, "BEFORE 1"
        CALL SLEEP(1)
        Z = 1
        !$OSS FLUSH
        PRINT *, "AFTER 1"
    !$OSS END TASK

    CALL FOO(V % ARR)

    !$OSS TASKWAIT

    PRINT *, "END"
END PROGRAM P

SUBROUTINE FOO(ARR)
    USE MOO, ONLY : Z
    IMPLICIT NONE
    INTEGER, POINTER :: ARR(:)

#ifdef __INTEL_COMPILER
    EXTERNAL :: ABORT
#endif

    PRINT *, "BEFORE 2"
    IF (Z /= 1) THEN
        PRINT *, "INVALID"
        CALL ABORT()
    END IF
    PRINT *, "AFTER 2"
END SUBROUTINE
