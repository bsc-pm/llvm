! This test checks lowering of OmpSs-2 DepOp.

! Test for subroutine

! RUN: flang-new -fc1 -fompss-2 -emit-llvm -fdisable-ompss-2-pass -flang-deprecated-no-hlfir %s -o - | FileCheck %s --check-prefix=LLVMIR

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

!LLVMIR-LABEL: define void @task_(ptr %0, ptr %1, ptr %2)
!LLVMIR:  %[[VAR_I:.*]] = alloca i32, i64 1, align 4
!LLVMIR:  %[[VAR_J:.*]] = alloca i32, i64 1, align 4
!LLVMIR:  %[[VLA_EXTENT:.*]] = select i1 %{{.*}}, i64 %{{.*}}, i64 0
!LLVMIR:  %[[VAR_ARRAY2:.*]] = alloca i32, i64 %{{.*}}, align 4

!LLVMIR: %[[VAR_X_LOAD:.*]] = load i32, ptr %0, align 4
!LLVMIR: %[[X_PLUS_6:.*]] = add i32 %[[VAR_X_LOAD]], 6
!LLVMIR: %[[X_PLUS_6_64:.*]] = sext i32 %[[X_PLUS_6]] to i64
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00"),
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %[[VAR_ARRAY2]], ptr undef),
!LLVMIR-SAME: "QUAL.OSS.VLA.DIMS"(ptr %[[VAR_ARRAY2]], i64 %[[VLA_EXTENT]]),
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %[[VAR_ARRAY2]], [11 x i8] c"dep string\00", ptr @compute.dep0, i64 %[[X_PLUS_6_64]], ptr %[[VAR_ARRAY2]]), "QUAL.OSS.CAPTURED"(i64 %[[VLA_EXTENT]], i64 4, i64 %[[X_PLUS_6_64]], i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %1, ptr undef)
!LLVMIR-SAME: "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAR_I]], i32 undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %1, [11 x i8] c"dep string\00", ptr @compute.dep1, ptr %1, ptr %[[VAR_I]]), "QUAL.OSS.CAPTURED"(i64 undef, i32 1, i32 10, i32 1) ]

!LLVMIR: %[[VAR_X_LOAD_0:.*]] = load i32, ptr %0, align 4
!LLVMIR: %[[X_PLUS_6_0:.*]] = add i32 %[[VAR_X_LOAD_0]], 6
!LLVMIR: %[[X_PLUS_6_64_0:.*]] = sext i32 %[[X_PLUS_6_0]] to i64
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %[[VAR_ARRAY2]], ptr undef)
!LLVMIR-SAME: "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAR_I]], i32 undef),
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %[[VAR_ARRAY2]], [11 x i8] c"dep string\00", ptr @compute.dep2, i64 %[[X_PLUS_6_64_0]], ptr %[[VAR_ARRAY2]], ptr %[[VAR_I]]), "QUAL.OSS.CAPTURED"(i64 %[[VLA_EXTENT]], i64 4, i64 %[[X_PLUS_6_64_0]], i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %1, ptr undef)
!LLVMIR-SAME: "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAR_I]], i32 undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %1, [11 x i8] c"dep string\00", ptr @compute.dep3, ptr %1, ptr %[[VAR_I]]), "QUAL.OSS.CAPTURED"(i64 undef, i32 1, i32 10, i32 1) ]

!LLVMIR: %[[VAR_X_LOAD_1:.*]] = load i32, ptr %0, align 4
!LLVMIR: %[[X_PLUS_6_1:.*]] = add i32 %[[VAR_X_LOAD_1]], 6
!LLVMIR: %[[X_PLUS_6_64_1:.*]] = sext i32 %[[X_PLUS_6_1]] to i64
!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %[[VAR_ARRAY2]], ptr undef)
!LLVMIR-SAME: "QUAL.OSS.VLA.DIMS"(ptr %[[VAR_ARRAY2]], i64 %[[VLA_EXTENT]])
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %[[VAR_ARRAY2]], [11 x i8] c"dep string\00", ptr @compute.dep4, i64 %[[X_PLUS_6_64_1]], ptr %[[VAR_ARRAY2]]), "QUAL.OSS.CAPTURED"(i64 %[[VLA_EXTENT]], i64 4, i64 %[[X_PLUS_6_64_1]], i32 1, i32 10, i32 1) ]

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep0(i64 %0, ptr %1)
!LLVMIR:   %3 = add i64 %0, -3
!LLVMIR-NEXT:   %4 = mul i64 %3, 4
!LLVMIR-NEXT:   %5 = insertvalue { ptr, i64, i64, i64 } undef, ptr %1, 0
!LLVMIR-NEXT:   %6 = insertvalue { ptr, i64, i64, i64 } %5, i64 %4, 1
!LLVMIR-NEXT:   %7 = insertvalue { ptr, i64, i64, i64 } %6, i64 0, 2
!LLVMIR-NEXT:   %8 = insertvalue { ptr, i64, i64, i64 } %7, i64 %4, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %8

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

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep2(i64 %0, ptr %1, ptr %2)
!LLVMIR:   %4 = add i64 %0, -3
!LLVMIR-NEXT:   %5 = icmp sgt i64 %4, 0
!LLVMIR-NEXT:   %6 = select i1 %5, i64 %4, i64 0
!LLVMIR-NEXT:   %7 = load i32, ptr %2, align 4
!LLVMIR-NEXT:   %8 = sext i32 %7 to i64
!LLVMIR-NEXT:   %9 = sub i64 %8, 4
!LLVMIR-NEXT:   %10 = add i64 %8, -3
!LLVMIR-NEXT:   %11 = mul i64 %6, 4
!LLVMIR-NEXT:   %12 = mul i64 %9, 4
!LLVMIR-NEXT:   %13 = mul i64 %10, 4
!LLVMIR-NEXT:   %14 = insertvalue { ptr, i64, i64, i64 } undef, ptr %1, 0
!LLVMIR-NEXT:   %15 = insertvalue { ptr, i64, i64, i64 } %14, i64 %11, 1
!LLVMIR-NEXT:   %16 = insertvalue { ptr, i64, i64, i64 } %15, i64 %12, 2
!LLVMIR-NEXT:   %17 = insertvalue { ptr, i64, i64, i64 } %16, i64 %13, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %17

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

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep4(i64 %0, ptr %1)
!LLVMIR:   %3 = add i64 %0, -3
!LLVMIR-NEXT:   %4 = icmp sgt i64 %3, 0
!LLVMIR-NEXT:   %5 = select i1 %4, i64 %3, i64 0
!LLVMIR-NEXT:   %6 = mul i64 %5, 4
!LLVMIR-NEXT:   %7 = mul i64 %3, 4
!LLVMIR-NEXT:   %8 = insertvalue { ptr, i64, i64, i64 } undef, ptr %1, 0
!LLVMIR-NEXT:   %9 = insertvalue { ptr, i64, i64, i64 } %8, i64 %6, 1
!LLVMIR-NEXT:   %10 = insertvalue { ptr, i64, i64, i64 } %9, i64 0, 2
!LLVMIR-NEXT:   %11 = insertvalue { ptr, i64, i64, i64 } %10, i64 %7, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %11
