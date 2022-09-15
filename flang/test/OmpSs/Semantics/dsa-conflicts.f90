! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

PROGRAM P1
    TYPE TY
        INTEGER :: X(10)
    END TYPE

    INTEGER :: ARRAY(10)
    INTEGER, POINTER :: P

    INTEGER :: I
    TYPE(TY) :: T

    ! OK
    !$OSS TASK SHARED(I) DEPEND(IN: I)
        CONTINUE
    !$OSS END TASK

    ! OK, promotion
    !$OSS TASK SHARED(I) DEPEND(IN: ARRAY(I))
        CONTINUE
    !$OSS END TASK

    ! KO, promotion but firstprivate afterwards
    !ERROR: data-sharing mismatch 'OSSShared' vs 'OSSFirstPrivate'
    !$OSS TASK DEPEND(IN: ARRAY(I)) SHARED(I) FIRSTPRIVATE(I)
        CONTINUE
    !$OSS END TASK

    ! KO, promotion but firstprivate afterwards
    !ERROR: data-sharing mismatch 'OSSShared' vs 'OSSFirstPrivate'
    !$OSS TASK DEPEND(IN: ARRAY(I), I) FIRSTPRIVATE(I)
        CONTINUE
    !$OSS END TASK

    ! KO, pointers have to be FIRSTPRIVATE
    !ERROR: data-sharing mismatch 'OSSShared' vs 'OSSFirstPrivate'
    !$OSS TASK SHARED(P) DEPEND(IN: P)
        CONTINUE
    !$OSS END TASK

    ! KO, pointers have to be FIRSTPRIVATE
    !ERROR: data-sharing mismatch 'OSSFirstPrivate' vs 'OSSShared'
    !$OSS TASK DEPEND(IN: P) SHARED(P)
        CONTINUE
    !$OSS END TASK

    ! KO, dependency needs SHARED
    !ERROR: data-sharing mismatch 'OSSFirstPrivate' vs 'OSSShared'
    !$OSS TASK FIRSTPRIVATE(I) DEPEND(IN: I)
        CONTINUE
    !$OSS END TASK

    ! OK
    !$OSS TASK FIRSTPRIVATE(I) DEPEND(IN: ARRAY(I))
        CONTINUE
    !$OSS END TASK

    ! OK
    !$OSS TASK SHARED(I) DEPEND(IN: T%X(I))
        CONTINUE
    !$OSS END TASK


END PROGRAM
