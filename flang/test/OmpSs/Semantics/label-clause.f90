! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2 

real function f()
    f = 1.0
end

PROGRAM P1
    CHARACTER(LEN=20) STR

    !$OSS TASK LABEL(STR)
        CONTINUE
    !$OSS END TASK

    !ERROR: Must have CHARACTER type, but is INTEGER(4)
    !$OSS TASK LABEL(1)
        CONTINUE
    !$OSS END TASK

    !$OSS TASK LABEL("HOLA")
        CONTINUE
    !$OSS END TASK

END PROGRAM
