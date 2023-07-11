! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

PROGRAM P1
    CHARACTER(LEN=20) :: C

    !ERROR: expected variable
    !$OSS TASK SHARED(C(4:7))
        CONTINUE
    !$OSS END TASK

END PROGRAM
