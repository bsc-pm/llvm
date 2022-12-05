! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

PROGRAM P
INTEGER :: X
INTEGER :: Y
INTEGER :: ARRAY(10, 10)

!$OSS TASK DO
DO X=0,10
DO Y=0,10
if (Y .eq. 7) EXIT
ARRAY(X,Y) = X + Y
!Y = Y + 1
END DO
END DO
!$OSS END TASK DO

END PROGRAM

! This tests checks allocas inside the directive

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR:  br label %oss.taskfor.region, !dbg !10
!LLVMIR: oss.taskfor.region:                               ; preds = %0
!LLVMIR:   %{{.*}} = alloca i32, i64 1, align 4, !dbg !12

