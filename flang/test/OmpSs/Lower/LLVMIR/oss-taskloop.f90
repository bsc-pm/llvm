! This test checks lowering of OmpSs-2 taskloop Directive.

! RUN: flang-new -fc1 -fompss-2 -emit-llvm -fdisable-ompss-2-pass %s -o - | FileCheck %s --check-prefix=LLVMIR

program task
    INTEGER :: I
    INTEGER :: J

    ! grainsize clause
    !$OSS TASKLOOP GRAINSIZE(50)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! No clauses
    !$OSS TASKLOOP
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! if clause
    !$OSS TASKLOOP IF(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! final clause
    !$OSS TASKLOOP FINAL(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! cost clause
    !$OSS TASKLOOP COST(3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! priority clause
    !$OSS TASKLOOP PRIORITY(3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! default clause
    !$OSS TASKLOOP DEFAULT(SHARED)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! private clause
    !$OSS TASKLOOP PRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! firstprivate clause
    !$OSS TASKLOOP FIRSTPRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! shared clause
    !$OSS TASKLOOP SHARED(I)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

end program


!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.LOOP.GRAINSIZE"(i32 50), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 50) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.IF"(i1 %6), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.FINAL"(i1 %9), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.COST"(i32 3), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 3) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIORITY"(i32 3), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 3) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.SHARED"(ptr %2, i32 undef), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]

