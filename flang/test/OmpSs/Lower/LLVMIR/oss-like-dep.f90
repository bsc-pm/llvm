! This test checks lowering of OmpSs-2 like dependencies.

! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program task
    IMPLICIT NONE
    INTEGER :: I

    !$OSS TASK OUT(I)
    CONTINUE
    !$OSS END TASK
end

!LLVMIR-LABEL: define void @_QQmain()
!LLVMIR: %[[VAR_I:.*]] = alloca i32, i64 1, align 4
!LLVMIR-NEXT: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(ptr %[[VAR_I]], i32 undef), "QUAL.OSS.DEP.OUT"(ptr %[[VAR_I]], [11 x i8] c"dep string\00", ptr @compute.dep0, ptr %[[VAR_I]]) ]

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep0(ptr %0)
!LLVMIR:   %2 = insertvalue { ptr, i64, i64, i64 } undef, ptr %0, 0
!LLVMIR-NEXT:   %3 = insertvalue { ptr, i64, i64, i64 } %2, i64 4, 1
!LLVMIR-NEXT:   %4 = insertvalue { ptr, i64, i64, i64 } %3, i64 0, 2
!LLVMIR-NEXT:   %5 = insertvalue { ptr, i64, i64, i64 } %4, i64 4, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %5

