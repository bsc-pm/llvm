! This test checks lowering of OmpSs-2 DepOp.

! Test for subroutine

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

! Support list
! - [x] assumed-size array
! - [x] assumed-shape array

subroutine task(X, ARRAY, ARRAY1)
    IMPLICIT NONE

    INTEGER :: X
    INTEGER :: ARRAY(*)
    INTEGER :: ARRAY1(:)
    INTEGER :: ARRAY2(4: x + 6)

    INTEGER :: I
    INTEGER :: J

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: ARRAY1)
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY2)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: ARRAY1(I))
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY2(I))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I : I + 10))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: ARRAY1(I : I + 10))
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    ! TODO
    !$OSS TASK DO DEPEND(OUT: ARRAY2( : ))
    DO J=1,10
    END DO
    !$OSS END TASK DO

end subroutine

!LLVMIR-LABEL: define void @_QPtask(ptr %0, ptr %1, ptr %2)
!LLVMIR:  %[[VAR_I:.*]] = alloca i32, i64 1, align 4
!LLVMIR:  %[[VAR_J:.*]] = alloca i32, i64 1, align 4
!LLVMIR:  %[[VLA_EXTENT:.*]] = select i1 %{{.*}}, i64 %{{.*}}, i64 0
!LLVMIR:  %[[VAR_ARRAY2:.*]] = alloca i32, i64 %{{.*}}, align 4

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"),
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %[[VAR_ARRAY2]], i32 undef),
!LLVMIR-SAME: "QUAL.OSS.VLA.DIMS"(ptr %[[VAR_ARRAY2]], i64 %[[VLA_EXTENT]]),
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %[[VAR_ARRAY2]], [11 x i8] c"dep string\00", ptr @compute.dep0, ptr %0, ptr %[[VAR_ARRAY2]]), "QUAL.OSS.CAPTURED"(i64 %[[VLA_EXTENT]], i64 4, i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %1, i32 undef)
!LLVMIR-SAME: "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAR_I]], i32 undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %1, [11 x i8] c"dep string\00", ptr @compute.dep1, ptr %1, ptr %[[VAR_I]]), "QUAL.OSS.CAPTURED"(i64 undef, i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %[[VAR_ARRAY2]], i32 undef)
!LLVMIR-SAME: "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAR_I]], i32 undef),
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %[[VAR_ARRAY2]], [11 x i8] c"dep string\00", ptr @compute.dep2, ptr %0, ptr %[[VAR_ARRAY2]], ptr %[[VAR_I]]), "QUAL.OSS.CAPTURED"(i64 %[[VLA_EXTENT]], i64 4, i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %1, i32 undef)
!LLVMIR-SAME: "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAR_I]], i32 undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %1, [11 x i8] c"dep string\00", ptr @compute.dep3, ptr %1, ptr %[[VAR_I]]), "QUAL.OSS.CAPTURED"(i64 undef, i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %[[VAR_ARRAY2]], i32 undef)
!LLVMIR-SAME: "QUAL.OSS.VLA.DIMS"(ptr %[[VAR_ARRAY2]], i64 %[[VLA_EXTENT]])
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %[[VAR_ARRAY2]], [11 x i8] c"dep string\00", ptr @compute.dep4, ptr %0, ptr %[[VAR_ARRAY2]]), "QUAL.OSS.CAPTURED"(i64 %[[VLA_EXTENT]], i64 4, i32 1, i32 10, i32 1) ]

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep0(ptr %0, ptr %1)
!LLVMIR-NEXT:   %3 = load i32, ptr %0, align 4
!LLVMIR-NEXT:   %4 = add i32 %3, 6
!LLVMIR-NEXT:   %5 = sext i32 %4 to i64
!LLVMIR-NEXT:   %6 = add i64 %5, -3
!LLVMIR-NEXT:   %7 = mul i64 %6, 4
!LLVMIR-NEXT:   %8 = insertvalue { ptr, i64, i64, i64 } undef, ptr %1, 0
!LLVMIR-NEXT:   %9 = insertvalue { ptr, i64, i64, i64 } %8, i64 %7, 1
!LLVMIR-NEXT:   %10 = insertvalue { ptr, i64, i64, i64 } %9, i64 0, 2
!LLVMIR-NEXT:   %11 = insertvalue { ptr, i64, i64, i64 } %10, i64 %7, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %11

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep1(ptr %0, ptr %1)
!LLVMIR:   %3 = load i32, ptr %1, align 4
!LLVMIR-NEXT:   %4 = sext i32 %3 to i64
!LLVMIR-NEXT:   %5 = sub i64 %4, 1
!LLVMIR-NEXT:   %6 = mul i64 %5, 4
!LLVMIR-NEXT:   %7 = mul i64 %4, 4
!LLVMIR-NEXT:   %8 = insertvalue { ptr, i64, i64, i64 } undef, ptr %0, 0
!LLVMIR-NEXT:   %9 = insertvalue { ptr, i64, i64, i64 } %8, i64 4, 1
!LLVMIR-NEXT:   %10 = insertvalue { ptr, i64, i64, i64 } %9, i64 %6, 2
!LLVMIR-NEXT:   %11 = insertvalue { ptr, i64, i64, i64 } %10, i64 %7, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %11

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep2(ptr %0, ptr %1, ptr %2)
!LLVMIR:   %4 = load i32, ptr %0, align 4
!LLVMIR-NEXT:   %5 = add i32 %4, 6
!LLVMIR-NEXT:   %6 = sext i32 %5 to i64
!LLVMIR-NEXT:   %7 = add i64 %6, -3
!LLVMIR-NEXT:   %8 = icmp sgt i64 %7, 0
!LLVMIR-NEXT:   %9 = select i1 %8, i64 %7, i64 0
!LLVMIR-NEXT:   %10 = load i32, ptr %2, align 4
!LLVMIR-NEXT:   %11 = sext i32 %10 to i64
!LLVMIR-NEXT:   %12 = sub i64 %11, 4
!LLVMIR-NEXT:   %13 = add i64 %11, -3
!LLVMIR-NEXT:   %14 = mul i64 %9, 4
!LLVMIR-NEXT:   %15 = mul i64 %12, 4
!LLVMIR-NEXT:   %16 = mul i64 %13, 4
!LLVMIR-NEXT:   %17 = insertvalue { ptr, i64, i64, i64 } undef, ptr %1, 0
!LLVMIR-NEXT:   %18 = insertvalue { ptr, i64, i64, i64 } %17, i64 %14, 1
!LLVMIR-NEXT:   %19 = insertvalue { ptr, i64, i64, i64 } %18, i64 %15, 2
!LLVMIR-NEXT:   %20 = insertvalue { ptr, i64, i64, i64 } %19, i64 %16, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %20

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep3(ptr %0, ptr %1)
!LLVMIR:   %3 = load i32, ptr %1, align 4
!LLVMIR-NEXT:   %4 = sext i32 %3 to i64
!LLVMIR-NEXT:   %5 = add i32 %3, 10
!LLVMIR-NEXT:   %6 = sext i32 %5 to i64
!LLVMIR-NEXT:   %7 = sub i64 %6, %4
!LLVMIR-NEXT:   %8 = add i64 %7, 1
!LLVMIR-NEXT:   %9 = icmp sgt i64 %8, 0
!LLVMIR-NEXT:   %10 = select i1 %9, i64 %8, i64 0
!LLVMIR-NEXT:   %11 = sub i64 %4, 1
!LLVMIR-NEXT:   %12 = mul i64 %10, 4
!LLVMIR-NEXT:   %13 = mul i64 %11, 4
!LLVMIR-NEXT:   %14 = mul i64 %6, 4
!LLVMIR-NEXT:   %15 = insertvalue { ptr, i64, i64, i64 } undef, ptr %0, 0
!LLVMIR-NEXT:   %16 = insertvalue { ptr, i64, i64, i64 } %15, i64 %12, 1
!LLVMIR-NEXT:   %17 = insertvalue { ptr, i64, i64, i64 } %16, i64 %13, 2
!LLVMIR-NEXT:   %18 = insertvalue { ptr, i64, i64, i64 } %17, i64 %14, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %18

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep4(ptr %0, ptr %1)
!LLVMIR:   %3 = load i32, ptr %0, align 4
!LLVMIR-NEXT:   %4 = add i32 %3, 6
!LLVMIR-NEXT:   %5 = sext i32 %4 to i64
!LLVMIR-NEXT:   %6 = add i64 %5, -3
!LLVMIR-NEXT:   %7 = icmp sgt i64 %6, 0
!LLVMIR-NEXT:   %8 = select i1 %7, i64 %6, i64 0
!LLVMIR-NEXT:   %9 = mul i64 %8, 4
!LLVMIR-NEXT:   %10 = mul i64 %6, 4
!LLVMIR-NEXT:   %11 = insertvalue { ptr, i64, i64, i64 } undef, ptr %1, 0
!LLVMIR-NEXT:   %12 = insertvalue { ptr, i64, i64, i64 } %11, i64 %9, 1
!LLVMIR-NEXT:   %13 = insertvalue { ptr, i64, i64, i64 } %12, i64 0, 2
!LLVMIR-NEXT:   %14 = insertvalue { ptr, i64, i64, i64 } %13, i64 %10, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %14
