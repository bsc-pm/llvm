! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

! Check all the default keywords

PROGRAM P1

    !$OSS TASK DEFAULT(SHARED)
        CONTINUE
    !$OSS END TASK

    !$OSS TASK DEFAULT(PRIVATE)
        CONTINUE
    !$OSS END TASK

    !$OSS TASK DEFAULT(FIRSTPRIVATE)
        CONTINUE
    !$OSS END TASK

END PROGRAM
