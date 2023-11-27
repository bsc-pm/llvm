! This test checks lowering of OmpSs-2 DepOp.

! Test for program

! RUN: flang-new -fc1 -fompss-2 -emit-llvm -fdisable-ompss-2-pass -flang-deprecated-no-hlfir %s -o - | FileCheck %s --check-prefix=LLVMIR

! Support list
! - [x] scalar
! - [x] explicit-shape array (no region)
! - [x] explicit-shape array (region)
! - [ ] assumed-size array
! - [ ] assumed-shape array
! - [ ] deferred-shape array

program task
    INTEGER :: J

    INTEGER :: I
    INTEGER :: ARRAY(10)
    INTEGER :: ARRAY1(5:10)
    TYPE TY
       INTEGER :: X
       INTEGER :: ARRAY(10)
    END TYPE
    TYPE(TY) T

    !$OSS TASK DO DEPEND(OUT: I)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: T%X)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! TODO
    !$OSS TASK DO DEPEND(OUT: T%ARRAY)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! TODO
    ! !$OSS TASK DO DEPEND(OUT: T%ARRAY(2))
    ! DO J=1,10
    ! END DO
    ! !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY1)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(I : I + 10))
    DO J=1,10
    END DO
    !$OSS END TASK DO

    !$OSS TASK DO DEPEND(OUT: ARRAY(:))
    DO J=1,10
    END DO
    !$OSS END TASK DO

end program

!LLVMIR-LABEL: define void @_QQmain()
!LLVMIR:  %[[VAR_ARRAY1:.*]] = alloca [6 x i32], i64 1, align 4
!LLVMIR:  %[[VAR_I:.*]] = alloca i32, i64 1, align 4

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %[[VAR_I]], i32 undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %[[VAR_I]], [11 x i8] c"dep string\00", ptr @compute.dep0, ptr %[[VAR_I]]), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr @_QFEarray, [10 x i32] undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr @_QFEarray, [11 x i8] c"dep string\00", ptr @compute.dep1, ptr @_QFEarray), "QUAL.OSS.CAPTURED"(i64 10, i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr @_QFEarray, [10 x i32] undef)
!LLVMIR-SAME: "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAR_I]], i32 undef),
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr @_QFEarray, [11 x i8] c"dep string\00", ptr @compute.dep2, ptr @_QFEarray, ptr %[[VAR_I]]), "QUAL.OSS.CAPTURED"(i64 10, i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr @_QFEt, %_QFTty undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr @_QFEt, [11 x i8] c"dep string\00", ptr @compute.dep3, ptr @_QFEt), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr @_QFEt, %_QFTty undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr @_QFEt, [11 x i8] c"dep string\00", ptr @compute.dep4, ptr @_QFEt), "QUAL.OSS.CAPTURED"(i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr %[[VAR_ARRAY1]], [6 x i32] undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr %[[VAR_ARRAY1]], [11 x i8] c"dep string\00", ptr @compute.dep5, ptr %[[VAR_ARRAY1]]), "QUAL.OSS.CAPTURED"(i64 6, i64 5, i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr @_QFEarray, [10 x i32] undef)
!LLVMIR-SAME: "QUAL.OSS.FIRSTPRIVATE"(ptr %[[VAR_I]], i32 undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr @_QFEarray, [11 x i8] c"dep string\00", ptr @compute.dep6, ptr @_QFEarray, ptr %[[VAR_I]]), "QUAL.OSS.CAPTURED"(i64 10, i32 1, i32 10, i32 1) ]

!LLVMIR: %{{.*}} = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASK.FOR\00")
!LLVMIR-SAME: "QUAL.OSS.SHARED"(ptr @_QFEarray, [10 x i32] undef)
!LLVMIR-SAME: "QUAL.OSS.DEP.OUT"(ptr @_QFEarray, [11 x i8] c"dep string\00", ptr @compute.dep7, ptr @_QFEarray), "QUAL.OSS.CAPTURED"(i64 10, i32 1, i32 10, i32 1) ]

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep0(ptr %0)
!LLVMIR:   %2 = insertvalue { ptr, i64, i64, i64 } undef, ptr %0, 0
!LLVMIR-NEXT:   %3 = insertvalue { ptr, i64, i64, i64 } %2, i64 4, 1
!LLVMIR-NEXT:   %4 = insertvalue { ptr, i64, i64, i64 } %3, i64 0, 2
!LLVMIR-NEXT:   %5 = insertvalue { ptr, i64, i64, i64 } %4, i64 4, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %5

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep1(ptr %0)
!LLVMIR:   %2 = insertvalue { ptr, i64, i64, i64 } undef, ptr %0, 0
!LLVMIR-NEXT:   %3 = insertvalue { ptr, i64, i64, i64 } %2, i64 40, 1
!LLVMIR-NEXT:   %4 = insertvalue { ptr, i64, i64, i64 } %3, i64 0, 2
!LLVMIR-NEXT:   %5 = insertvalue { ptr, i64, i64, i64 } %4, i64 40, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %5

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep2(ptr %0, ptr %1)
!LLVMIR:   %3 = load i32, ptr %1, align 4
!LLVMIR-NEXT:   %4 = sext i32 %3 to i64
!LLVMIR-NEXT:   %5 = sub i64 %4, 1
!LLVMIR-NEXT:   %6 = mul i64 %5, 4
!LLVMIR-NEXT:   %7 = mul i64 %4, 4
!LLVMIR-NEXT:   %8 = insertvalue { ptr, i64, i64, i64 } undef, ptr %0, 0
!LLVMIR-NEXT:   %9 = insertvalue { ptr, i64, i64, i64 } %8, i64 40, 1
!LLVMIR-NEXT:   %10 = insertvalue { ptr, i64, i64, i64 } %9, i64 %6, 2
!LLVMIR-NEXT:   %11 = insertvalue { ptr, i64, i64, i64 } %10, i64 %7, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %11

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep3(ptr %0)
!LLVMIR:   %2 = getelementptr %_QFTty, ptr %0, i32 0, i32 0
!LLVMIR-NEXT:   %3 = insertvalue { ptr, i64, i64, i64 } undef, ptr %2, 0
!LLVMIR-NEXT:   %4 = insertvalue { ptr, i64, i64, i64 } %3, i64 4, 1
!LLVMIR-NEXT:   %5 = insertvalue { ptr, i64, i64, i64 } %4, i64 0, 2
!LLVMIR-NEXT:   %6 = insertvalue { ptr, i64, i64, i64 } %5, i64 4, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %6

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep4(ptr %0)
!LLVMIR:   %2 = getelementptr %_QFTty, ptr %0, i32 0, i32 1
!LLVMIR-NEXT:   %3 = insertvalue { ptr, i64, i64, i64 } undef, ptr %2, 0
!LLVMIR-NEXT:   %4 = insertvalue { ptr, i64, i64, i64 } %3, i64 40, 1
!LLVMIR-NEXT:   %5 = insertvalue { ptr, i64, i64, i64 } %4, i64 0, 2
!LLVMIR-NEXT:   %6 = insertvalue { ptr, i64, i64, i64 } %5, i64 40, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %6

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep5(ptr %0)
!LLVMIR:   %2 = insertvalue { ptr, i64, i64, i64 } undef, ptr %0, 0
!LLVMIR-NEXT:   %3 = insertvalue { ptr, i64, i64, i64 } %2, i64 24, 1
!LLVMIR-NEXT:   %4 = insertvalue { ptr, i64, i64, i64 } %3, i64 0, 2
!LLVMIR-NEXT:   %5 = insertvalue { ptr, i64, i64, i64 } %4, i64 24, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %5

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep6(ptr %0, ptr %1)
!LLVMIR:   %3 = load i32, ptr %1, align 4
!LLVMIR-NEXT:   %4 = sext i32 %3 to i64
!LLVMIR-NEXT:   %5 = add i32 %3, 10
!LLVMIR-NEXT:   %6 = sext i32 %5 to i64
!LLVMIR-NEXT:   %7 = sub i64 %4, 1
!LLVMIR-NEXT:   %8 = mul i64 %7, 4
!LLVMIR-NEXT:   %9 = mul i64 %6, 4
!LLVMIR-NEXT:   %10 = insertvalue { ptr, i64, i64, i64 } undef, ptr %0, 0
!LLVMIR-NEXT:   %11 = insertvalue { ptr, i64, i64, i64 } %10, i64 40, 1
!LLVMIR-NEXT:   %12 = insertvalue { ptr, i64, i64, i64 } %11, i64 %8, 2
!LLVMIR-NEXT:   %13 = insertvalue { ptr, i64, i64, i64 } %12, i64 %9, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %13

!LLVMIR-LABEL: define { ptr, i64, i64, i64 } @compute.dep7(ptr %0)
!LLVMIR:   %2 = insertvalue { ptr, i64, i64, i64 } undef, ptr %0, 0
!LLVMIR-NEXT:   %3 = insertvalue { ptr, i64, i64, i64 } %2, i64 40, 1
!LLVMIR-NEXT:   %4 = insertvalue { ptr, i64, i64, i64 } %3, i64 0, 2
!LLVMIR-NEXT:   %5 = insertvalue { ptr, i64, i64, i64 } %4, i64 40, 3
!LLVMIR-NEXT:   ret { ptr, i64, i64, i64 } %5
