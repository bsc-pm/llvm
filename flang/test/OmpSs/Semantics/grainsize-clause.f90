! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

real function f()
    f = 1.0
end

PROGRAM P1
    INTEGER :: I
    LOGICAL, DIMENSION (2) :: B

    !ERROR: Must have INTEGER type, but is LOGICAL(4)
    !$OSS TASKLOOP GRAINSIZE(B)
    DO I = 1, 10
        CONTINUE
    END DO
    !$OSS END TASKLOOP

    !ERROR: The parameter of the GRAINSIZE clause must be a positive integer expression
    !$OSS TASKLOOP GRAINSIZE(-1)
    DO I = 1, 10
        CONTINUE
    END DO
    !$OSS END TASKLOOP

    !ERROR: Must have INTEGER type, but is REAL(4)
    !$OSS TASKLOOP GRAINSIZE(f())
    DO I = 1, 10
        CONTINUE
    END DO
    !$OSS END TASKLOOP

END PROGRAM
