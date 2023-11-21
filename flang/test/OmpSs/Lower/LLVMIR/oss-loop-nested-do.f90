! This test checks lowering of OmpSs-2 loop Directives.
! All induction variables inside the construct are private

! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

subroutine task()
    INTEGER :: I
    INTEGER :: J

    !$OSS TASK
    DO I=1,10
        DO J=1,10
            CONTINUE
        END DO
    END DO
    !$OSS END TASK
end

subroutine taskfor()
    INTEGER :: I
    INTEGER :: J

    !$OSS TASK DO
    DO I=1,10
        DO J=1,10
            CONTINUE
        END DO
    END DO
    !$OSS END TASK DO
end

subroutine taskloop()
    INTEGER :: I
    INTEGER :: J

    !$OSS TASKLOOP
    DO I=1,10
        DO J=1,10
            CONTINUE
        END DO
    END DO
    !$OSS END TASKLOOP
end

subroutine taskloopfor()
    INTEGER :: I
    INTEGER :: J

    !$OSS TASKLOOP DO
    DO I=1,10
        DO J=1,10
            CONTINUE
        END DO
    END DO
    !$OSS END TASKLOOP DO
end


!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 1), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.TYPE"(i64 1, i64 1, i64 1, i64 1, i64 1), "QUAL.OSS.LOOP.IND.VAR"(ptr %1), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef), "QUAL.OSS.PRIVATE"(ptr %2, i32 undef), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]

