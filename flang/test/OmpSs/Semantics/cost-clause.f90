! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2 

real function f()
    f = 1.0
end

PROGRAM P1
    INTEGER :: I 
    LOGICAL, DIMENSION (2) :: B

    !ERROR: Must have INTEGER type, but is LOGICAL(4)
    !$OSS TASK COST(B)
        CONTINUE
    !$OSS END TASK

    !ERROR: The parameter of the COST clause must be a positive integer expression
    !$OSS TASK COST(-1)
        CONTINUE
    !$OSS END TASK

    !ERROR: Must have INTEGER type, but is REAL(4)
    !$OSS TASK COST(f())
        CONTINUE
    !$OSS END TASK

END PROGRAM
