! This test checks lowering of OmpSs-2 loop Directives.
! All induction variables inside the construct are private

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

subroutine task()
    INTEGER :: I
    INTEGER :: J

    !$OSS TASK
    DO I=1,10
        DO J=1,10
            CONTINUE
        END DO
    END DO
    !$OSS END TASK
end

subroutine taskfor()
    INTEGER :: I
    INTEGER :: J

    !$OSS TASK DO
    DO I=1,10
        DO J=1,10
            CONTINUE
        END DO
    END DO
    !$OSS END TASK DO
end

subroutine taskloop()
    INTEGER :: I
    INTEGER :: J

    !$OSS TASKLOOP
    DO I=1,10
        DO J=1,10
            CONTINUE
        END DO
    END DO
    !$OSS END TASKLOOP
end

subroutine taskloopfor()
    INTEGER :: I
    INTEGER :: J

    !$OSS TASKLOOP DO
    DO I=1,10
        DO J=1,10
            CONTINUE
        END DO
    END DO
    !$OSS END TASKLOOP DO
end

!FIRDialect-LABEL: func @_QPtask()
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtaskEi"}
!FIRDialect-NEXT:  %[[VAR_J:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtaskEj"}
!FIRDialect-NEXT:  oss.task private(%[[VAR_I]], %[[VAR_J]] : !fir.ref<i32>, !fir.ref<i32>)

!FIRDialect-LABEL: func @_QPtaskfor()
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtaskforEi"}
!FIRDialect-NEXT:  %[[VAR_J:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtaskforEj"}
!FIRDialect-NEXT:  %[[LB_0:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_0:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_0:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_0:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_0]] : i32) upper_bound(%[[UB_0]] : i32) step(%[[STEP_0]] : i32) loop_type(%[[LTYPE_0]] : i64) ind_var(%[[VAR_I]] : !fir.ref<i32>) private(%[[VAR_I]], %[[VAR_J]] : !fir.ref<i32>, !fir.ref<i32>)

!FIRDialect-LABEL: func @_QPtaskloop()
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtaskloopEi"}
!FIRDialect-NEXT:  %[[VAR_J:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtaskloopEj"}
!FIRDialect-NEXT:  %[[LB_0:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_0:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_0:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_0:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.taskloop lower_bound(%[[LB_0]] : i32) upper_bound(%[[UB_0]] : i32) step(%[[STEP_0]] : i32) loop_type(%[[LTYPE_0]] : i64) ind_var(%[[VAR_I]] : !fir.ref<i32>) private(%[[VAR_I]], %[[VAR_J]] : !fir.ref<i32>, !fir.ref<i32>)

!FIRDialect-LABEL: func @_QPtaskloopfor()
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtaskloopforEi"}
!FIRDialect-NEXT:  %[[VAR_J:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtaskloopforEj"}
!FIRDialect-NEXT:  %[[LB_0:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_0:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_0:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_0:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.taskloop_for lower_bound(%[[LB_0]] : i32) upper_bound(%[[UB_0]] : i32) step(%[[STEP_0]] : i32) loop_type(%[[LTYPE_0]] : i64) ind_var(%[[VAR_I]] : !fir.ref<i32>) private(%[[VAR_I]], %[[VAR_J]] : !fir.ref<i32>, !fir.ref<i32>)

