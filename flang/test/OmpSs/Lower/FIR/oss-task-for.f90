! This test checks lowering of OmpSs-2 task do Directive.

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program task
    INTEGER :: I
    INTEGER :: J

    ! No clauses
    !$OSS TASK DO
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! if clause
    !$OSS TASK DO IF(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! final clause
    !$OSS TASK DO FINAL(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! cost clause
    !$OSS TASK DO COST(3)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! priority clause
    !$OSS TASK DO PRIORITY(3)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! default clause
    !$OSS TASK DO DEFAULT(SHARED)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! private clause
    !$OSS TASK DO PRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! firstprivate clause
    !$OSS TASK DO FIRSTPRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! shared clause
    !$OSS TASK DO SHARED(I)
    DO J=1,10
    END DO
    !$OSS END TASK DO

    ! chunksize clause
    !$OSS TASK DO CHUNKSIZE(3)
    DO J=1,10
    END DO
    !$OSS END TASK DO

end program

!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
!FIRDialect-NEXT:  %[[VAR_J:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
!FIRDialect-NEXT:  %[[LB_0:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_0:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_0:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_0:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_0]] : i32) upper_bound(%[[UB_0]] : i32) step(%[[STEP_0]] : i32) loop_type(%[[LTYPE_0]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) private(%[[VAR_J]] : !fir.ref<i32>)
!FIRDialect-NEXT:    oss.terminator

!FIRDialect:  %[[LOAD_VAR_I_0:.*]] = fir.load %[[VAR_I]] : !fir.ref<i32>
!FIRDialect-NEXT:  %[[CONSTANT_3_0:.*]] = arith.constant 3 : i32
!FIRDialect-NEXT:  %[[CMP_0:.*]] = arith.cmpi eq, %[[LOAD_VAR_I_0]], %[[CONSTANT_3_0]] : i32
!FIRDialect-NEXT:  %[[LB_1:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_1:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_1:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_1:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_1]] : i32) upper_bound(%[[UB_1]] : i32) step(%[[STEP_1]] : i32) loop_type(%[[LTYPE_1]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) if(%[[CMP_0]] : i1) private(%[[VAR_J]] : !fir.ref<i32>)

!FIRDialect:  %[[LOAD_VAR_I_1:.*]] = fir.load %[[VAR_I]] : !fir.ref<i32>
!FIRDialect-NEXT:  %[[CONSTANT_3_1:.*]] = arith.constant 3 : i32
!FIRDialect-NEXT:  %[[CMP_1:.*]] = arith.cmpi eq, %[[LOAD_VAR_I_1]], %[[CONSTANT_3_1]] : i32
!FIRDialect-NEXT:  %[[LB_2:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_2:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_2:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_2:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_2]] : i32) upper_bound(%[[UB_2]] : i32) step(%[[STEP_2]] : i32) loop_type(%[[LTYPE_2]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) final(%[[CMP_1]] : i1) private(%[[VAR_J]] : !fir.ref<i32>)

!FIRDialect:  %[[CONSTANT_3_2:.*]] = arith.constant 3 : i32
!FIRDialect-NEXT:  %[[LB_3:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_3:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_3:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_3:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_3]] : i32) upper_bound(%[[UB_3]] : i32) step(%[[STEP_3]] : i32) loop_type(%[[LTYPE_3]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) cost(%[[CONSTANT_3_2]] : i32) private(%[[VAR_J]] : !fir.ref<i32>)

!FIRDialect:  %[[CONSTANT_3_3:.*]] = arith.constant 3 : i32
!FIRDialect-NEXT:  %[[LB_4:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_4:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_4:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_4:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_4]] : i32) upper_bound(%[[UB_4]] : i32) step(%[[STEP_4]] : i32) loop_type(%[[LTYPE_4]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) priority(%[[CONSTANT_3_3]] : i32) private(%[[VAR_J]] : !fir.ref<i32>)

!FIRDialect:  %[[LB_5:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_5:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_5:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_5:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_5]] : i32) upper_bound(%[[UB_5]] : i32) step(%[[STEP_5]] : i32) loop_type(%[[LTYPE_5]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) private(%[[VAR_J]] : !fir.ref<i32>) default( defshared)

!FIRDialect:  %[[LB_6:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_6:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_6:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_6:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_6]] : i32) upper_bound(%[[UB_6]] : i32) step(%[[STEP_6]] : i32) loop_type(%[[LTYPE_6]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) private(%[[VAR_I]], %[[VAR_J]] : !fir.ref<i32>, !fir.ref<i32>)

!FIRDialect:  %[[LB_7:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_7:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_7:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_7:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_7]] : i32) upper_bound(%[[UB_7]] : i32) step(%[[STEP_7]] : i32) loop_type(%[[LTYPE_7]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) private(%[[VAR_J]] : !fir.ref<i32>) firstprivate(%[[VAR_I]] : !fir.ref<i32>)

!FIRDialect:  %[[LB_8:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_8:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_8:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_8:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_8]] : i32) upper_bound(%[[UB_8]] : i32) step(%[[STEP_8]] : i32) loop_type(%[[LTYPE_8]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) private(%[[VAR_J]] : !fir.ref<i32>) shared(%[[VAR_I]] : !fir.ref<i32>)

!FIRDialect:  %[[CONSTANT_3_4:.*]] = arith.constant 3 : i32
!FIRDialect-NEXT:  %[[LB_9:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[UB_9:.*]] = arith.constant 10 : i32
!FIRDialect-NEXT:  %[[STEP_9:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[LTYPE_9:.*]] = arith.constant 1 : i64
!FIRDialect-NEXT:  oss.task_for lower_bound(%[[LB_9]] : i32) upper_bound(%[[UB_9]] : i32) step(%[[STEP_9]] : i32) loop_type(%[[LTYPE_9]] : i64) ind_var(%[[VAR_J]] : !fir.ref<i32>) private(%[[VAR_J]] : !fir.ref<i32>) chunksize(%[[CONSTANT_3_4]] : i32)

