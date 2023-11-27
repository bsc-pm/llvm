! This test checks lowering of OmpSs-2 release Directive.

! Test for program

! RUN: flang-new -fc1 -fompss-2 -emit-llvm -fdisable-ompss-2-pass -flang-deprecated-no-hlfir %s -o - | FileCheck %s --check-prefix=LLVMIR

program release
    IMPLICIT NONE
    INTEGER :: I

    !$OSS RELEASE IN(I)
end

!LLVMIR-LABEL: define void @_QQmain()
!LLVMIR:  %[[VAR_I:.*]] = alloca i32, i64 1, align 4
!LLVMIR-NEXT: %{{.*}} = call i1 @llvm.directive.marker() [ "DIR.OSS"([8 x i8] c"RELEASE\00"), "QUAL.OSS.DEP.IN"(ptr %1, [11 x i8] c"dep string\00", ptr @compute.dep0, ptr %1) ]

