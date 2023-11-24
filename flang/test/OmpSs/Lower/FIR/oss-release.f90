! This test checks lowering of OmpSs-2 release Directive.

! Test for program

! RUN: flang-new -fc1 -emit-fir -fompss-2 -o - %s -flang-deprecated-no-hlfir | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program release
    IMPLICIT NONE
    INTEGER :: I

    !$OSS RELEASE IN(I)
end

!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
!FIRDialect-NEXT:  %[[DEP_0:.*]] = oss.dependency base(%[[VAR_I]] : !fir.ref<i32>) function(@compute.dep0) arguments(%[[VAR_I]] : !fir.ref<i32>) -> i32
!FIRDialect-NEXT:  oss.release in(%[[DEP_0]] : i32)

