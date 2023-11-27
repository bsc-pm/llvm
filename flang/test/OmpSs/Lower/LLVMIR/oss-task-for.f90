! This test checks lowering of OmpSs-2 task do Directive.

! RUN: flang-new -fc1 -fompss-2 -emit-llvm -fdisable-ompss-2-pass %s -o - | FileCheck %s --check-prefix=LLVMIR

program task
    INTEGER :: I
    INTEGER :: J

    ! chunksize clause
    !$OSS TASK DO CHUNKSIZE(50)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! No clauses
    !$OSS TASK DO
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! if clause
    !$OSS TASK DO IF(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! final clause
    !$OSS TASK DO FINAL(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! cost clause
    !$OSS TASK DO COST(3)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! priority clause
    !$OSS TASK DO PRIORITY(3)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! default clause
    !$OSS TASK DO DEFAULT(SHARED)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! private clause
    !$OSS TASK DO PRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! firstprivate clause
    !$OSS TASK DO FIRSTPRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! shared clause
    !$OSS TASK DO SHARED(I)
    DO J=1,10
    END DO
    !$OSS END TASK DO

end program



!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.LOOP.CHUNKSIZE"(i32 50), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 50) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.IF"(i1 %6), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.FINAL"(i1 %9), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.COST"(i32 3), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 3) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.PRIORITY"(i32 3), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1, i32 3) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.FIRSTPRIVATE"(ptr %1, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %2), "QUAL.OSS.SHARED"(ptr %1, i32 undef), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]

