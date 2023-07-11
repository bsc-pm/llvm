! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

SUBROUTINE S(X)
  INTEGER :: X(*)

  ! OK: Implicit DSA SHARED
  !$OSS TASK
  X(1) = 4
  !$OSS END TASK

  ! OK: Explicit DSA SHARED
  !$OSS TASK SHARED(X)
  X(1) = 4
  !$OSS END TASK

  ! KO: Assumed-size arrays are SHARED
  !ERROR: Assumed size array cannot be OSSPrivate
  !$OSS TASK PRIVATE(X)
  X(1) = 4
  !$OSS END TASK
END SUBROUTINE

PROGRAM P1
    TYPE TY
        INTEGER X
    END TYPE

    INTEGER :: COARRAY[*]
    INTEGER :: ARRAY(10)

    INTEGER :: I
    TYPE(TY) :: T
    COMMON /C/ I

    INTEGER, PARAMETER :: O = 5

    !$OSS TASK SHARED(I, I)
        CONTINUE
    !$OSS END TASK

    !ERROR: data-sharing mismatch 'OSSShared' vs 'OSSPrivate'
    !$OSS TASK SHARED(I) PRIVATE(I)
        CONTINUE
    !$OSS END TASK

    !ERROR: data-sharing mismatch 'OSSPrivate' vs 'OSSFirstPrivate'
    !$OSS TASK PRIVATE(I) FIRSTPRIVATE(I)
        CONTINUE
    !$OSS END TASK

    !ERROR: data-sharing mismatch 'OSSShared' vs 'OSSFirstPrivate'
    !$OSS TASK SHARED(I) FIRSTPRIVATE(I)
        CONTINUE
    !$OSS END TASK

    !ERROR: COMMON is not supported yet
    !$OSS TASK SHARED(/C/)
        CONTINUE
    !$OSS END TASK

    !ERROR: expected variable
    !$OSS TASK SHARED(T%X)
        CONTINUE
    !$OSS END TASK

    !ERROR: expected variable
    !$OSS TASK SHARED(COARRAY[3])
        CONTINUE
    !$OSS END TASK

    !ERROR: expected variable
    !$OSS TASK SHARED(ARRAY[3])
        CONTINUE
    !$OSS END TASK

    ! KO, PARAMETER is not valid in DSA clauses
    !ERROR: expected variable
    !$OSS TASK SHARED(O)
        CONTINUE
    !$OSS END TASK

END PROGRAM
