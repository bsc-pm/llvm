! NOTE: Assertions have been autogenerated by /home/rpenacob/llvm-mono/mlir/utils/generate-test-checks.py
! This test checks lowering of OmpSs-2 task do Directive.

! RUN: flang-new -fc1 -emit-fir -fompss-2 -o - %s | FileCheck %s --check-prefix=FIRDialect

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


! FIRDialect-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "task"} {
! FIRDialect:           %[[VAL_0:[-0-9A-Za-z._]+]] = arith.constant 3 : i32
! FIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = arith.constant 1 : i64
! FIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = arith.constant 10 : i32
! FIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = arith.constant 1 : i32
! FIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! FIRDialect:           %[[VAL_5:[-0-9A-Za-z._]+]] = fir.declare %[[VAL_4]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
! FIRDialect:           %[[VAL_6:[-0-9A-Za-z._]+]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
! FIRDialect:           %[[VAL_7:[-0-9A-Za-z._]+]] = fir.declare %[[VAL_6]] {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
! FIRDialect:           %[[VAL_8:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_9:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) private(%[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_8]], %[[VAL_9]] : !fir.oss<i32>, !fir.oss<i32>) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_10:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_11:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_12:[-0-9A-Za-z._]+]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! FIRDialect:           %[[VAL_13:[-0-9A-Za-z._]+]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_0]] : i32
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) if(%[[VAL_13]] : i1) private(%[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_10]], %[[VAL_11]] : !fir.oss<i32>, !fir.oss<i32>) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_14:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_15:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_16:[-0-9A-Za-z._]+]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! FIRDialect:           %[[VAL_17:[-0-9A-Za-z._]+]] = arith.cmpi eq, %[[VAL_16]], %[[VAL_0]] : i32
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) final(%[[VAL_17]] : i1) private(%[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_14]], %[[VAL_15]] : !fir.oss<i32>, !fir.oss<i32>) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_18:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_19:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) cost(%[[VAL_0]] : i32) private(%[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_18]], %[[VAL_19]] : !fir.oss<i32>, !fir.oss<i32>) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_20:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_21:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) priority(%[[VAL_0]] : i32) private(%[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_20]], %[[VAL_21]] : !fir.oss<i32>, !fir.oss<i32>) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_22:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_23:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) private(%[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_22]], %[[VAL_23]] : !fir.oss<i32>, !fir.oss<i32>) default( defshared) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_24:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_25:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_26:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_27:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) private(%[[VAL_5]], %[[VAL_5]], %[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_27]] : !fir.oss<i32>, !fir.oss<i32>, !fir.oss<i32>, !fir.oss<i32>) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_28:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_29:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_30:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_31:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) private(%[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_28]], %[[VAL_29]] : !fir.oss<i32>, !fir.oss<i32>) firstprivate(%[[VAL_5]], %[[VAL_5]] : !fir.ref<i32>, !fir.ref<i32>) firstprivate_type(%[[VAL_30]], %[[VAL_31]] : !fir.oss<i32>, !fir.oss<i32>) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_32:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_33:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_34:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_35:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) private(%[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_34]], %[[VAL_35]] : !fir.oss<i32>, !fir.oss<i32>) shared(%[[VAL_5]], %[[VAL_5]] : !fir.ref<i32>, !fir.ref<i32>) shared_type(%[[VAL_32]], %[[VAL_33]] : !fir.oss<i32>, !fir.oss<i32>) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_36:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           %[[VAL_37:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task_for lower_bound(%[[VAL_3]] : i32) upper_bound(%[[VAL_2]] : i32) step(%[[VAL_3]] : i32) loop_type(%[[VAL_1]] : i64) ind_var(%[[VAL_7]] : !fir.ref<i32>) private(%[[VAL_7]], %[[VAL_7]] : !fir.ref<i32>, !fir.ref<i32>) private_type(%[[VAL_36]], %[[VAL_37]] : !fir.oss<i32>, !fir.oss<i32>) chunksize(%[[VAL_0]] : i32) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           return
! FIRDialect:         }

