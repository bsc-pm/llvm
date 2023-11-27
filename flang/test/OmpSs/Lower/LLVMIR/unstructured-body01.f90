! RUN: flang-new -fc1 -fompss-2 -emit-llvm -fdisable-ompss-2-pass -flang-deprecated-no-hlfir %s -o - | FileCheck %s --check-prefix=LLVMIR

PROGRAM P
INTEGER :: X
INTEGER :: Y
INTEGER :: ARRAY(10, 10)

!$OSS TASK
DO X=0,10
DO Y=0,10
if (Y .eq. 7) EXIT
ARRAY(X,Y) = X + Y
END DO
END DO
!$OSS END TASK

END PROGRAM

! This tests checks allocas inside the directive

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00")
!LLVMIR:  br label %oss.task.region
!LLVMIR: oss.task.region:                                  ; preds = %0
!LLVMIR:  %{{.*}} = alloca i32, i64 1, align 4
!LLVMIR:  %{{.*}} = alloca i32, i64 1, align 4
