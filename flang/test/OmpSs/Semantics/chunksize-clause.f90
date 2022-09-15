! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

real function f()
    f = 1.0
end

PROGRAM P1
    INTEGER :: I
    LOGICAL, DIMENSION (2) :: B

    !ERROR: Must have INTEGER type, but is LOGICAL(4)
    !$OSS TASK DO CHUNKSIZE(B)
    DO I = 1, 10
        CONTINUE
    END DO
    !$OSS END TASK DO

    !ERROR: The parameter of the CHUNKSIZE clause must be a positive integer expression
    !$OSS TASK DO CHUNKSIZE(-1)
    DO I = 1, 10
        CONTINUE
    END DO
    !$OSS END TASK DO

    !ERROR: Must have INTEGER type, but is REAL(4)
    !$OSS TASK DO CHUNKSIZE(f())
    DO I = 1, 10
        CONTINUE
    END DO
    !$OSS END TASK DO

END PROGRAM
