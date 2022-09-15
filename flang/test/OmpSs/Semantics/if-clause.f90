! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2 

real function f()
    f = 1.0
end

PROGRAM P1
    INTEGER :: I 
    LOGICAL, DIMENSION (2) :: B

    !ERROR: Must be a scalar value, but is a rank-1 array
    !$OSS TASK IF(B)
        CONTINUE
    !$OSS END TASK

    !ERROR: Must have LOGICAL type, but is INTEGER(4)
    !$OSS TASK IF(I)
        CONTINUE
    !$OSS END TASK

    !ERROR: Must have LOGICAL type, but is REAL(4)
    !$OSS TASK IF(f())
        CONTINUE
    !$OSS END TASK

END PROGRAM
