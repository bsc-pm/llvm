! This test Canonicalize does not perform optimizations between values
! that are defined outside the task and used inside.

! RUN: bbc -hlfir=false -fompss-2 %s -o - | \
! RUN:   fir-opt --canonicalize 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

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

!LLVMIRDialect-LABEL: func @_QQmain()
!LLVMIRDialect:  %[[FOO_VAR:.*]] = fir.alloca
!LLVMIRDialect:  oss.task firstprivate(%[[FOO_VAR]] : !fir.ref<!fir.type<_QFTfoo_t{a:i32}>>)
!LLVMIRDialect-NEXT:    %[[CONSTANT_1:.*]] = arith.constant 1 : i32
!LLVMIRDialect-NEXT:    %[[FIELD_0:.*]] = fir.field_index a, !fir.type<_QFTfoo_t{a:i32}>
!LLVMIRDialect-NEXT:    %[[COORD_0:.*]] = fir.coordinate_of %[[FOO_VAR]], %[[FIELD_0]] : (!fir.ref<!fir.type<_QFTfoo_t{a:i32}>>, !fir.field) -> !fir.ref<i32>
!LLVMIRDialect-NEXT:    fir.store %[[CONSTANT_1]] to %[[COORD_0]] : !fir.ref<i32>
!LLVMIRDialect-NEXT:    oss.terminator

