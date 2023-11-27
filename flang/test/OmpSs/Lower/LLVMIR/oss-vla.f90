! This test checks lowering of OmpSs-2 VlaDimOp.

! Test for subroutine

! RUN: flang-new -fc1 -fompss-2 -emit-llvm -fdisable-ompss-2-pass -flang-deprecated-no-hlfir %s -o - | FileCheck %s --check-prefix=LLVMIR

subroutine task(X)
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY1(1: x + 6,1:3)
    INTEGER :: ARRAY2(1: x + 6,2:3)
    INTEGER :: ARRAY3(4: x + 6,1:3)
    INTEGER :: ARRAY4(4: x + 6,2:3)

    INTEGER :: I
    INTEGER :: J

    !$OSS TASK DO
    DO J=1,3
      DO I=1,x+6
        ARRAY1(I,J) = (I+J)
      END DO
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO
    DO J=2,3
      DO I=1,x+6
        ARRAY2(I,J) = I+J
      END DO
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO
    DO J=1,3
      DO I=4,x+6
        ARRAY3(I,J) = I+J
      END DO
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO
    DO J=2,3
      DO I=4,x+6
        ARRAY4(I,J) = I+J
      END DO
    END DO
    !$OSS END TASK DO

end subroutine

!LLVMIR-LABEL: define void @task_(ptr %0)


!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVM-SAME: "QUAL.OSS.VLA.DIMS"(ptr %10, i64 %8, i64 3)
!LLVM-SAME: "QUAL.OSS.CAPTURED"(i64 %8, i64 3, i32 1, i32 3, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVM-SAME: "QUAL.OSS.VLA.DIMS"(ptr %12, i64 %8, i64 2)
!LLVM-SAME: "QUAL.OSS.CAPTURED"(i64 %8, i64 2, i64 1, i64 2, i32 2, i32 3, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVM-SAME: "QUAL.OSS.VLA.DIMS"(ptr %17, i64 %15, i64 3)
!LLVM-SAME: "QUAL.OSS.CAPTURED"(i64 %15, i64 3, i64 4, i64 1, i32 1, i32 3, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVM-SAME: "QUAL.OSS.VLA.DIMS"(ptr %19, i64 %15, i64 2)
!LLVM-SAME: "QUAL.OSS.CAPTURED"(i64 %15, i64 2, i64 4, i64 2, i32 2, i32 3, i32 1) ]

