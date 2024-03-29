! NOTE: Assertions have been autogenerated by /home/rpenacob/llvm-mono/mlir/utils/generate-test-checks.py
! This test checks lowering of OmpSs-2 loop Directives.
! All induction variables inside the construct are private

! RUN: bbc -hlfir=false -fompss-2 %s -o - | fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 |  FileCheck %s --check-prefix=LLVMIRDialect

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


! LLVMIRDialect-LABEL:   llvm.func @_QPtask() {
! LLVMIRDialect:           %[[VAL_0:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_0]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
! LLVMIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_2]] x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
! LLVMIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = llvm.mlir.undef : i32
! LLVMIRDialect:           oss.task private(%[[VAL_1]], %[[VAL_3]] : !llvm.ptr, !llvm.ptr) private_type(%[[VAL_4]], %[[VAL_4]] : i32, i32) {
! LLVMIRDialect:             %[[VAL_5:[-0-9A-Za-z._]+]] = llvm.mlir.constant(0 : index) : i64
! LLVMIRDialect:             %[[VAL_6:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : index) : i64
! LLVMIRDialect:             %[[VAL_7:[-0-9A-Za-z._]+]] = llvm.mlir.constant(10 : index) : i64
! LLVMIRDialect:             %[[VAL_8:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:             %[[VAL_9:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_8]] x i32 {bindc_name = "j", pinned} : (i64) -> !llvm.ptr
! LLVMIRDialect:             %[[VAL_10:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:             %[[VAL_11:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_10]] x i32 {bindc_name = "i", pinned} : (i64) -> !llvm.ptr
! LLVMIRDialect:             %[[VAL_12:[-0-9A-Za-z._]+]] = llvm.trunc %[[VAL_6]] : i64 to i32
! LLVMIRDialect:             llvm.br ^bb1(%[[VAL_12]], %[[VAL_7]] : i32, i64)
! LLVMIRDialect:           ^bb1(%[[VAL_13:[-0-9A-Za-z._]+]]: i32, %[[VAL_14:[-0-9A-Za-z._]+]]: i64):
! LLVMIRDialect:             %[[VAL_15:[-0-9A-Za-z._]+]] = llvm.icmp "sgt" %[[VAL_14]], %[[VAL_5]] : i64
! LLVMIRDialect:             llvm.cond_br %[[VAL_15]], ^bb2, ^bb6
! LLVMIRDialect:           ^bb2:
! LLVMIRDialect:             llvm.store %[[VAL_13]], %[[VAL_11]] : i32, !llvm.ptr
! LLVMIRDialect:             llvm.br ^bb3(%[[VAL_12]], %[[VAL_7]] : i32, i64)
! LLVMIRDialect:           ^bb3(%[[VAL_16:[-0-9A-Za-z._]+]]: i32, %[[VAL_17:[-0-9A-Za-z._]+]]: i64):
! LLVMIRDialect:             %[[VAL_18:[-0-9A-Za-z._]+]] = llvm.icmp "sgt" %[[VAL_17]], %[[VAL_5]] : i64
! LLVMIRDialect:             llvm.cond_br %[[VAL_18]], ^bb4, ^bb5
! LLVMIRDialect:           ^bb4:
! LLVMIRDialect:             llvm.store %[[VAL_16]], %[[VAL_9]] : i32, !llvm.ptr
! LLVMIRDialect:             %[[VAL_19:[-0-9A-Za-z._]+]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i32
! LLVMIRDialect:             %[[VAL_20:[-0-9A-Za-z._]+]] = llvm.add %[[VAL_19]], %[[VAL_12]]  : i32
! LLVMIRDialect:             %[[VAL_21:[-0-9A-Za-z._]+]] = llvm.sub %[[VAL_17]], %[[VAL_6]]  : i64
! LLVMIRDialect:             llvm.br ^bb3(%[[VAL_20]], %[[VAL_21]] : i32, i64)
! LLVMIRDialect:           ^bb5:
! LLVMIRDialect:             llvm.store %[[VAL_16]], %[[VAL_9]] : i32, !llvm.ptr
! LLVMIRDialect:             %[[VAL_22:[-0-9A-Za-z._]+]] = llvm.load %[[VAL_11]] : !llvm.ptr -> i32
! LLVMIRDialect:             %[[VAL_23:[-0-9A-Za-z._]+]] = llvm.add %[[VAL_22]], %[[VAL_12]]  : i32
! LLVMIRDialect:             %[[VAL_24:[-0-9A-Za-z._]+]] = llvm.sub %[[VAL_14]], %[[VAL_6]]  : i64
! LLVMIRDialect:             llvm.br ^bb1(%[[VAL_23]], %[[VAL_24]] : i32, i64)
! LLVMIRDialect:           ^bb6:
! LLVMIRDialect:             llvm.store %[[VAL_13]], %[[VAL_11]] : i32, !llvm.ptr
! LLVMIRDialect:             oss.terminator
! LLVMIRDialect:           }
! LLVMIRDialect:           llvm.return
! LLVMIRDialect:         }

! LLVMIRDialect-LABEL:   llvm.func @_QPtaskfor() {
! LLVMIRDialect:           %[[VAL_0:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = llvm.mlir.constant(10 : i32) : i32
! LLVMIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i32) : i32
! LLVMIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_3]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
! LLVMIRDialect:           %[[VAL_5:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_6:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_5]] x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
! LLVMIRDialect:           %[[VAL_7:[-0-9A-Za-z._]+]] = llvm.mlir.undef : i32
! LLVMIRDialect:           oss.task_for lower_bound(%[[VAL_2]] : i32) upper_bound(%[[VAL_1]] : i32) step(%[[VAL_2]] : i32) loop_type(%[[VAL_0]] : i64) ind_var(%[[VAL_4]] : !llvm.ptr) private(%[[VAL_4]], %[[VAL_6]] : !llvm.ptr, !llvm.ptr) private_type(%[[VAL_7]], %[[VAL_7]] : i32, i32) {
! LLVMIRDialect:             %[[VAL_8:[-0-9A-Za-z._]+]] = llvm.mlir.constant(0 : index) : i64
! LLVMIRDialect:             %[[VAL_9:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : index) : i64
! LLVMIRDialect:             %[[VAL_10:[-0-9A-Za-z._]+]] = llvm.mlir.constant(10 : index) : i64
! LLVMIRDialect:             %[[VAL_11:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:             %[[VAL_12:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_11]] x i32 {bindc_name = "j", pinned} : (i64) -> !llvm.ptr
! LLVMIRDialect:             %[[VAL_13:[-0-9A-Za-z._]+]] = llvm.trunc %[[VAL_9]] : i64 to i32
! LLVMIRDialect:             llvm.br ^bb1(%[[VAL_13]], %[[VAL_10]] : i32, i64)
! LLVMIRDialect:           ^bb1(%[[VAL_14:[-0-9A-Za-z._]+]]: i32, %[[VAL_15:[-0-9A-Za-z._]+]]: i64):
! LLVMIRDialect:             %[[VAL_16:[-0-9A-Za-z._]+]] = llvm.icmp "sgt" %[[VAL_15]], %[[VAL_8]] : i64
! LLVMIRDialect:             llvm.cond_br %[[VAL_16]], ^bb2, ^bb3
! LLVMIRDialect:           ^bb2:
! LLVMIRDialect:             llvm.store %[[VAL_14]], %[[VAL_12]] : i32, !llvm.ptr
! LLVMIRDialect:             %[[VAL_17:[-0-9A-Za-z._]+]] = llvm.load %[[VAL_12]] : !llvm.ptr -> i32
! LLVMIRDialect:             %[[VAL_18:[-0-9A-Za-z._]+]] = llvm.add %[[VAL_17]], %[[VAL_13]]  : i32
! LLVMIRDialect:             %[[VAL_19:[-0-9A-Za-z._]+]] = llvm.sub %[[VAL_15]], %[[VAL_9]]  : i64
! LLVMIRDialect:             llvm.br ^bb1(%[[VAL_18]], %[[VAL_19]] : i32, i64)
! LLVMIRDialect:           ^bb3:
! LLVMIRDialect:             llvm.store %[[VAL_14]], %[[VAL_12]] : i32, !llvm.ptr
! LLVMIRDialect:             oss.terminator
! LLVMIRDialect:           }
! LLVMIRDialect:           llvm.return
! LLVMIRDialect:         }

! LLVMIRDialect-LABEL:   llvm.func @_QPtaskloop() {
! LLVMIRDialect:           %[[VAL_0:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = llvm.mlir.constant(10 : i32) : i32
! LLVMIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i32) : i32
! LLVMIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_3]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
! LLVMIRDialect:           %[[VAL_5:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_6:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_5]] x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
! LLVMIRDialect:           %[[VAL_7:[-0-9A-Za-z._]+]] = llvm.mlir.undef : i32
! LLVMIRDialect:           oss.taskloop lower_bound(%[[VAL_2]] : i32) upper_bound(%[[VAL_1]] : i32) step(%[[VAL_2]] : i32) loop_type(%[[VAL_0]] : i64) ind_var(%[[VAL_4]] : !llvm.ptr) private(%[[VAL_4]], %[[VAL_6]] : !llvm.ptr, !llvm.ptr) private_type(%[[VAL_7]], %[[VAL_7]] : i32, i32) {
! LLVMIRDialect:             %[[VAL_8:[-0-9A-Za-z._]+]] = llvm.mlir.constant(0 : index) : i64
! LLVMIRDialect:             %[[VAL_9:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : index) : i64
! LLVMIRDialect:             %[[VAL_10:[-0-9A-Za-z._]+]] = llvm.mlir.constant(10 : index) : i64
! LLVMIRDialect:             %[[VAL_11:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:             %[[VAL_12:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_11]] x i32 {bindc_name = "j", pinned} : (i64) -> !llvm.ptr
! LLVMIRDialect:             %[[VAL_13:[-0-9A-Za-z._]+]] = llvm.trunc %[[VAL_9]] : i64 to i32
! LLVMIRDialect:             llvm.br ^bb1(%[[VAL_13]], %[[VAL_10]] : i32, i64)
! LLVMIRDialect:           ^bb1(%[[VAL_14:[-0-9A-Za-z._]+]]: i32, %[[VAL_15:[-0-9A-Za-z._]+]]: i64):
! LLVMIRDialect:             %[[VAL_16:[-0-9A-Za-z._]+]] = llvm.icmp "sgt" %[[VAL_15]], %[[VAL_8]] : i64
! LLVMIRDialect:             llvm.cond_br %[[VAL_16]], ^bb2, ^bb3
! LLVMIRDialect:           ^bb2:
! LLVMIRDialect:             llvm.store %[[VAL_14]], %[[VAL_12]] : i32, !llvm.ptr
! LLVMIRDialect:             %[[VAL_17:[-0-9A-Za-z._]+]] = llvm.load %[[VAL_12]] : !llvm.ptr -> i32
! LLVMIRDialect:             %[[VAL_18:[-0-9A-Za-z._]+]] = llvm.add %[[VAL_17]], %[[VAL_13]]  : i32
! LLVMIRDialect:             %[[VAL_19:[-0-9A-Za-z._]+]] = llvm.sub %[[VAL_15]], %[[VAL_9]]  : i64
! LLVMIRDialect:             llvm.br ^bb1(%[[VAL_18]], %[[VAL_19]] : i32, i64)
! LLVMIRDialect:           ^bb3:
! LLVMIRDialect:             llvm.store %[[VAL_14]], %[[VAL_12]] : i32, !llvm.ptr
! LLVMIRDialect:             oss.terminator
! LLVMIRDialect:           }
! LLVMIRDialect:           llvm.return
! LLVMIRDialect:         }

! LLVMIRDialect-LABEL:   llvm.func @_QPtaskloopfor() {
! LLVMIRDialect:           %[[VAL_0:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = llvm.mlir.constant(10 : i32) : i32
! LLVMIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i32) : i32
! LLVMIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_3]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
! LLVMIRDialect:           %[[VAL_5:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:           %[[VAL_6:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_5]] x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
! LLVMIRDialect:           %[[VAL_7:[-0-9A-Za-z._]+]] = llvm.mlir.undef : i32
! LLVMIRDialect:           oss.taskloop_for lower_bound(%[[VAL_2]] : i32) upper_bound(%[[VAL_1]] : i32) step(%[[VAL_2]] : i32) loop_type(%[[VAL_0]] : i64) ind_var(%[[VAL_4]] : !llvm.ptr) private(%[[VAL_4]], %[[VAL_6]] : !llvm.ptr, !llvm.ptr) private_type(%[[VAL_7]], %[[VAL_7]] : i32, i32) {
! LLVMIRDialect:             %[[VAL_8:[-0-9A-Za-z._]+]] = llvm.mlir.constant(0 : index) : i64
! LLVMIRDialect:             %[[VAL_9:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : index) : i64
! LLVMIRDialect:             %[[VAL_10:[-0-9A-Za-z._]+]] = llvm.mlir.constant(10 : index) : i64
! LLVMIRDialect:             %[[VAL_11:[-0-9A-Za-z._]+]] = llvm.mlir.constant(1 : i64) : i64
! LLVMIRDialect:             %[[VAL_12:[-0-9A-Za-z._]+]] = llvm.alloca %[[VAL_11]] x i32 {bindc_name = "j", pinned} : (i64) -> !llvm.ptr
! LLVMIRDialect:             %[[VAL_13:[-0-9A-Za-z._]+]] = llvm.trunc %[[VAL_9]] : i64 to i32
! LLVMIRDialect:             llvm.br ^bb1(%[[VAL_13]], %[[VAL_10]] : i32, i64)
! LLVMIRDialect:           ^bb1(%[[VAL_14:[-0-9A-Za-z._]+]]: i32, %[[VAL_15:[-0-9A-Za-z._]+]]: i64):
! LLVMIRDialect:             %[[VAL_16:[-0-9A-Za-z._]+]] = llvm.icmp "sgt" %[[VAL_15]], %[[VAL_8]] : i64
! LLVMIRDialect:             llvm.cond_br %[[VAL_16]], ^bb2, ^bb3
! LLVMIRDialect:           ^bb2:
! LLVMIRDialect:             llvm.store %[[VAL_14]], %[[VAL_12]] : i32, !llvm.ptr
! LLVMIRDialect:             %[[VAL_17:[-0-9A-Za-z._]+]] = llvm.load %[[VAL_12]] : !llvm.ptr -> i32
! LLVMIRDialect:             %[[VAL_18:[-0-9A-Za-z._]+]] = llvm.add %[[VAL_17]], %[[VAL_13]]  : i32
! LLVMIRDialect:             %[[VAL_19:[-0-9A-Za-z._]+]] = llvm.sub %[[VAL_15]], %[[VAL_9]]  : i64
! LLVMIRDialect:             llvm.br ^bb1(%[[VAL_18]], %[[VAL_19]] : i32, i64)
! LLVMIRDialect:           ^bb3:
! LLVMIRDialect:             llvm.store %[[VAL_14]], %[[VAL_12]] : i32, !llvm.ptr
! LLVMIRDialect:             oss.terminator
! LLVMIRDialect:           }
! LLVMIRDialect:           llvm.return
! LLVMIRDialect:         }

