! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

PROGRAM P1
    INTEGER :: I
    INTEGER :: ARRAY(10)
    CHARACTER(LEN=20) :: STR
    COMMON /C/ I

    !ERROR: COMMON is not allowed
    !$OSS TASK DEPEND(IN: /C/)
        CONTINUE
    !$OSS END TASK

    !ERROR: Substrings are not allowed on OmpSs-2 clauses
    !$OSS TASK DEPEND(IN: STR(2:4))
        CONTINUE
    !$OSS END TASK

    !$OSS TASK DEPEND(IN: ARRAY(2:4))
        CONTINUE
    !$OSS END TASK

END PROGRAM
