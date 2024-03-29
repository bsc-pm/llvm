! RUN: %flang_fc1 -fompss-2 -fdebug-unparse-no-sema %s | FileCheck %s

SUBROUTINE FOO()
END SUBROUTINE

PROGRAM P1
    INTEGER :: I
    INTEGER :: S, P, F
!$OSS TASKLOOP DO SHARED(s) PRIVATE(P) FIRSTPRIVATE(F) DEFAULT(SHARED)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO DEPEND(IN: S) DEPEND(OUT: P) DEPEND(INOUT: F)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO DEPEND(WEAK, IN: S) DEPEND(WEAK, OUT: P) DEPEND(WEAK, INOUT: F)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO DEPEND(IN, WEAK: S) DEPEND(OUT, WEAK: P) DEPEND(INOUT, WEAK: F)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO DEPEND(INOUTSET: S) DEPEND(MUTEXINOUTSET: P)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO DEPEND(WEAK, INOUTSET: S) DEPEND(WEAK, MUTEXINOUTSET: P)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO DEPEND(INOUTSET, WEAK: S) DEPEND(MUTEXINOUTSET, WEAK: P)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO REDUCTION(+: S)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO IF(.TRUE.) FINAL(.TRUE.) PRIORITY(1) COST(1)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO WAIT LABEL("T1")
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO CHUNKSIZE(1) GRAINSIZE(1)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO ONREADY(FOO())
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO
END PROGRAM

SUBROUTINE OSS_DEPS()
    INTEGER :: S, P, F

!$OSS TASKLOOP DO IN(S) OUT(P) INOUT(F)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO WEAKIN(S) WEAKOUT(P) WEAKINOUT(F)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO CONCURRENT(S) COMMUTATIVE(P)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

!$OSS TASKLOOP DO WEAKCONCURRENT(S) WEAKCOMMUTATIVE(P)
    DO I = 1, 10
        CONTINUE
    END DO
!$OSS END TASKLOOP DO

END SUBROUTINE

!CHECK: !$OSS TASKLOOP DO  SHARED(s) PRIVATE(p) FIRSTPRIVATE(f) DEFAULT(SHARED)
!CHECK: !$OSS TASKLOOP DO  DEPEND(IN:s) DEPEND(OUT:p) DEPEND(INOUT:f)
!CHECK: !$OSS TASKLOOP DO  DEPEND(WEAK, IN:s) DEPEND(WEAK, OUT:p) DEPEND(WEAK, INOUT:f&
!CHECK: !$OSS&)
!CHECK: !$OSS TASKLOOP DO  DEPEND(WEAK, IN:s) DEPEND(WEAK, OUT:p) DEPEND(WEAK, INOUT:f&
!CHECK: !$OSS&)
!CHECK: !$OSS TASKLOOP DO  DEPEND(INOUTSET:s) DEPEND(MUTEXINOUTSET:p)
!CHECK: !$OSS TASKLOOP DO  DEPEND(WEAK, INOUTSET:s) DEPEND(WEAK, MUTEXINOUTSET:p)
!CHECK: !$OSS TASKLOOP DO  DEPEND(WEAK, INOUTSET:s) DEPEND(WEAK, MUTEXINOUTSET:p)
!CHECK: !$OSS TASKLOOP DO  REDUCTION(+:s)
!CHECK: !$OSS TASKLOOP DO  IF(.TRUE.) FINAL(.TRUE.) PRIORITY(1) COST(1)
!CHECK: !$OSS TASKLOOP DO  WAIT LABEL("T1")
!CHECK: !$OSS TASKLOOP DO  CHUNKSIZE(1) GRAINSIZE(1)
!CHECK: !$OSS TASKLOOP DO  ONREADY(foo())
!CHECK: !$OSS TASKLOOP DO  IN(s) OUT(p) INOUT(f)
!CHECK: !$OSS TASKLOOP DO  WEAKIN(s) WEAKOUT(p) WEAKINOUT(f)
!CHECK: !$OSS TASKLOOP DO  CONCURRENT(s) COMMUTATIVE(p)
!CHECK: !$OSS TASKLOOP DO  WEAKCONCURRENT(s) WEAKCOMMUTATIVE(p)

