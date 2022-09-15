! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2 

real function f()
    f = 1.0
end

PROGRAM P1
    INTEGER :: I 
    LOGICAL, DIMENSION (2) :: B

    !ERROR: Must have INTEGER type, but is LOGICAL(4)
    !$OSS TASK PRIORITY(B)
        CONTINUE
    !$OSS END TASK

    !$OSS TASK PRIORITY(-1)
        CONTINUE
    !$OSS END TASK

    !ERROR: Must have INTEGER type, but is REAL(4)
    !$OSS TASK PRIORITY(f())
        CONTINUE
    !$OSS END TASK

END PROGRAM
