! Tests stmt-function-stmt symbols capture and building
! temporal allocas inside task body

! TODO: this test fails with since this comment has been added.
! It seems that N and M are not in the SymMap and we cannot bind
! them with our task symbol

! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
SUBROUTINE FOO(N, M)
    IMPLICIT NONE
    INTEGER :: N
    INTEGER :: M
    INTEGER :: RES

    INTEGER :: F, X, Y
    F(X, Y) = X + Y + N
    INTEGER :: Q
    Q(X) = X + M

    !$OSS TASK SHARED(RES)
        RES = F(10, 100)
        !$OSS TASK SHARED(RES)
            RES = Q(1000)
        !$OSS END TASK
    !$OSS END TASK

END SUBROUTINE FOO

!FIRDialect-LABEL: func @_QPfoo(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<i32> {fir.bindc_name = "m"})
!FIRDialect:  %[[RES:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFfooEres"}
!FIRDialect:  oss.task
!FIRDialect-SAME:  firstprivate(%arg0, %arg1 : !fir.ref<i32>, !fir.ref<i32>)
!FIRDialect-SAME:  shared(%[[RES]] : !fir.ref<i32>)
!FIRDialect-NEXT:    %[[TMP_0:.*]] = fir.alloca i32
!FIRDialect-NEXT:    %[[TMP_1:.*]] = fir.alloca i32
!FIRDialect:    oss.task
!FIRDialect-SAME:    firstprivate(%arg1 : !fir.ref<i32>)
!FIRDialect-SAME:    shared(%[[RES]] : !fir.ref<i32>)
!FIRDialect-NEXT:      %[[TMP_2:.*]] = fir.alloca i32

