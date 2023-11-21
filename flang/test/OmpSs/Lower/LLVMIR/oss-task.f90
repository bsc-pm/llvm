! This test checks lowering of OmpSs-2 task Directive.

! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program task
    INTEGER :: I

    ! No clauses
    !$OSS TASK
    !$OSS END TASK

    ! if clause
    !$OSS TASK IF(I .EQ. 3)
    !$OSS END TASK

    ! final clause
    !$OSS TASK FINAL(I .EQ. 3)
    !$OSS END TASK

    ! cost clause
    !$OSS TASK COST(3)
    !$OSS END TASK

    ! priority clause
    !$OSS TASK PRIORITY(3)
    !$OSS END TASK

    ! default clause
    !$OSS TASK DEFAULT(SHARED)
    !$OSS END TASK

    ! private clause
    !$OSS TASK PRIVATE(I)
    !$OSS END TASK

    ! firstprivate clause
    !$OSS TASK FIRSTPRIVATE(I)
    !$OSS END TASK

    ! shared clause
    !$OSS TASK SHARED(I)
    !$OSS END TASK

end program


!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.IF"(i1 %4) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FINAL"(i1 %7) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.COST"(i32 3), "QUAL.OSS.CAPTURED"(i32 3) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIORITY"(i32 3), "QUAL.OSS.CAPTURED"(i32 3) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00") ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.PRIVATE"(ptr %1, i32 undef) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.FIRSTPRIVATE"(ptr %1, i32 undef) ]
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(ptr %1, i32 undef) ]

