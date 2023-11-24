! This test Canonicalize does not perform optimizations between values
! that are defined outside the task and used inside.

! RUN: flang-new -fc1 -emit-fir -fompss-2 -o - %s | \
! RUN:   fir-opt --canonicalize 2>&1 | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

PROGRAM P
    IMPLICIT NONE
    TYPE FOO_T
        INTEGER :: A
    END TYPE
    TYPE(FOO_T) :: FOO_VAR

    FOO_VAR % A = 0
    !$OSS TASK
    FOO_VAR % A = 1
    !$OSS END TASK

END PROGRAM P

!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[FOO_VAR:.*]] = fir.declare %{{.*}} {uniq_name = "_QFEfoo_var"}
!FIRDialect:  oss.task firstprivate(%[[FOO_VAR]], %[[FOO_VAR]] : !fir.ref<!fir.type<_QFTfoo_t{a:i32}>>, !fir.ref<!fir.type<_QFTfoo_t{a:i32}>>)
!FIRDialect-NEXT:    %[[CONSTANT_1:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:    %[[FIELD_0:.*]] = fir.field_index a, !fir.type<_QFTfoo_t{a:i32}>
!FIRDialect-NEXT:    %[[COORD_0:.*]] = fir.coordinate_of %[[FOO_VAR]], %[[FIELD_0]] : (!fir.ref<!fir.type<_QFTfoo_t{a:i32}>>, !fir.field) -> !fir.ref<i32>
!FIRDialect-NEXT:    fir.store %[[CONSTANT_1]] to %[[COORD_0]] : !fir.ref<i32>
!FIRDialect-NEXT:    oss.terminator

