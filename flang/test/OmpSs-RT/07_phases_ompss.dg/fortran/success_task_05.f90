! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>

MODULE M
IMPLICIT NONE
TYPE MATRIX_T
    INTEGER :: I
END TYPE

CONTAINS
    SUBROUTINE MAT_VEC(MAT)
        IMPLICIT NONE
        TYPE (MATRIX_T) :: MAT

        !$OSS TASK SHARED(MAT)
            MAT % I = 1
        !$OSS END TASK
    END SUBROUTINE MAT_VEC

END MODULE M

PROGRAM P
    USE M
    IMPLICIT NONE

    TYPE(MATRIX_T) :: MAT
    MAT % I = 0
    CALL MAT_VEC(MAT)
    !$OSS TASKWAIT
END PROGRAM P
