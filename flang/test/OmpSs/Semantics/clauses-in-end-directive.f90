! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

! It is not allowed to put clauses in END {TASK|TASK FOR...} directives

PROGRAM P1
    !$OSS TASK
        CONTINUE
    !ERROR: IF clause is not allowed on the END TASK directive
    !$OSS END TASK IF(.TRUE.)
END PROGRAM
