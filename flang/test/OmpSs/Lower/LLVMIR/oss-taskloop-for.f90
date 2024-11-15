! This test checks lowering of OmpSs-2 taskloop do Directive.

! RUN: flang-new -fc1 -fompss-2 -emit-llvm -fdisable-ompss-2-pass %s -o - | FileCheck %s --check-prefix=LLVMIR

program task
    INTEGER :: I
    INTEGER :: J

    ! chunksize clause
    !$OSS TASKLOOP DO CHUNKSIZE(50)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! grainsize clause
    !$OSS TASKLOOP DO GRAINSIZE(50)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! No clauses
    !$OSS TASKLOOP DO
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! if clause
    !$OSS TASKLOOP DO IF(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! final clause
    !$OSS TASKLOOP DO FINAL(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! cost clause
    !$OSS TASKLOOP DO COST(3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! priority clause
    !$OSS TASKLOOP DO PRIORITY(3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! default clause
    !$OSS TASKLOOP DO DEFAULT(SHARED)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! private clause
    !$OSS TASKLOOP DO PRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! firstprivate clause
    !$OSS TASKLOOP DO FIRSTPRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

    ! shared clause
    !$OSS TASKLOOP DO SHARED(I)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP DO

end program


!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.LOOP.CHUNKSIZE"(i32 50), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 50) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.LOOP.GRAINSIZE"(i32 50), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 50) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.IF"(i1 %7), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.FINAL"(i1 %10), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.COST"(i32 3), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 3) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIORITY"(i32 3), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 3) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.SHARED"(ptr %2, i32 undef), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]

