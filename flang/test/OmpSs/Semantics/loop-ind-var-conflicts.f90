! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

PROGRAM P1
    INTEGER :: I

    !ERROR: data-sharing mismatch 'OSSPrivate' vs 'OSSShared'
    !$OSS TASK DO SHARED(I)
    DO I=1,10
        CONTINUE
    ENDDO
    !$OSS END TASK DO

END PROGRAM
