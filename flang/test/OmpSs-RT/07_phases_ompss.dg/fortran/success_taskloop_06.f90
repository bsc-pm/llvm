! RUN: %oss-compile-and-run

! NOTE: checking results using a regular loop
! because it seems to be ANY does not work properly

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>
PROGRAM P
    IMPLICIT NONE
    INTEGER :: G, I
    INTEGER, PARAMETER :: MAX_NUM_TASKS = 20
    INTEGER, PARAMETER :: N = 2000
    INTEGER :: A(N)

    INTEGER :: L, U, S

    
    A = 0
    L = 1
    U = 2000
    S = 1
    DO G=1, MAX_NUM_TASKS

        !$OSS TASKLOOP SHARED(A) GRAINSIZE(G)
        DO I=L, U, S
            A(I) = A(I) + 1
        ENDDO
        !$OSS TASKWAIT
    ENDDO
!    IF(ANY(A(::S) /= MAX_NUM_TASKS)) STOP -1
    DO I=L, U, S
        IF(A(I) /= MAX_NUM_TASKS) STOP -1
    END DO


    A = 0
    L = 1
    U = 2000
    S = 3
    DO G=1, MAX_NUM_TASKS

        !$OSS TASKLOOP SHARED(A) GRAINSIZE(G)
        DO I=L, U, S
            A(I) = A(I) + 1
        ENDDO
        !$OSS TASKWAIT
    ENDDO
!    IF(ANY(A(::S) /= MAX_NUM_TASKS)) STOP -2
    DO I=L, U, S
        IF(A(I) /= MAX_NUM_TASKS) STOP -2
    END DO


    A = 0
    L = 2000
    U = 1
    S = -1
    DO G=1, MAX_NUM_TASKS

        !$OSS TASKLOOP SHARED(A) GRAINSIZE(G)
        DO I=L, U, S
            A(I) = A(I) + 1
        ENDDO
        !$OSS TASKWAIT
    ENDDO
!    IF(ANY(A(::S) /= MAX_NUM_TASKS)) STOP -3
    DO I=L, U, S
        IF(A(I) /= MAX_NUM_TASKS) STOP -3
    END DO


    A = 0
    L = 2000
    U = 1
    S = -3
    DO G=1, MAX_NUM_TASKS

        !$OSS TASKLOOP SHARED(A) GRAINSIZE(G)
        DO I=L, U, S
            A(I) = A(I) + 1
        ENDDO
        !$OSS TASKWAIT
    ENDDO
!    IF(ANY(A(::S) /= MAX_NUM_TASKS)) STOP -4
    DO I=L, U, S
        IF(A(I) /= MAX_NUM_TASKS) STOP -4
    END DO


    A = 0
    L = -999
    U = 1000
    S = 1
    DO G=1, MAX_NUM_TASKS

        !$OSS TASKLOOP SHARED(A) GRAINSIZE(G)
        DO I=L, U, S
            A(I+1000) = A(I+1000) + 1
        ENDDO
        !$OSS TASKWAIT
    ENDDO
!    IF(ANY(A(::S) /= MAX_NUM_TASKS)) STOP -5
    DO I=L, U, S
        IF(A(I+1000) /= MAX_NUM_TASKS) STOP -4
    END DO


    A = 0
    L = -999
    U = 1000
    S = 3
    DO G=1, MAX_NUM_TASKS

        !$OSS TASKLOOP SHARED(A) GRAINSIZE(G)
        DO I=L, U, S
            A(I+1000) = A(I+1000) + 1
        ENDDO
        !$OSS TASKWAIT
    ENDDO
!    IF(ANY(A(::S) /= MAX_NUM_TASKS)) STOP -6
    DO I=L, U, S
        IF(A(I+1000) /= MAX_NUM_TASKS) STOP -6
    END DO


    A = 0
    L = 999
    U = -1000
    S = -1
    DO G=1, MAX_NUM_TASKS

        !$OSS TASKLOOP SHARED(A) GRAINSIZE(G)
        DO I=L, U, S
            A(1000 - I) = A(1000 - I) + 1
        ENDDO
        !$OSS TASKWAIT
    ENDDO
!    IF(ANY(A(::S) /= MAX_NUM_TASKS)) STOP -7
    DO I=L, U, S
        IF(A(1000 - I) /= MAX_NUM_TASKS) STOP -7
    END DO


    A = 0
    L = 999
    U = -1000
    S = -3
    DO G=1, MAX_NUM_TASKS

        !$OSS TASKLOOP SHARED(A) GRAINSIZE(G)
        DO I=L, U, S
            A(1000 - I) = A(1000 - I) + 1
        ENDDO
        !$OSS TASKWAIT
    ENDDO
!    IF(ANY(A(::S) /= MAX_NUM_TASKS)) STOP -8
    DO I=L, U, S
        IF(A(1000 - I) /= MAX_NUM_TASKS) STOP -8
    END DO

END PROGRAM P
