! This test checks lowering of OmpSs-2 loop Directives.
! All induction variables inside the construct are private

! RUN: bbc -fompss-2 %s -o - | \
! RUN:   fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

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

!LLVMIRDialect-LABEL: llvm.func @_QPtask()
!LLVMIRDialect:  %[[CONSTANT_5:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_I:.*]] = llvm.alloca %[[CONSTANT_5]]
!LLVMIRDialect-NEXT:  %[[CONSTANT_6:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_J:.*]] = llvm.alloca %[[CONSTANT_6]]
!LLVMIRDialect-NEXT:  oss.task private(%[[VAR_I]], %[[VAR_J]] : !llvm.ptr<i32>, !llvm.ptr<i32>)

!LLVMIRDialect-LABEL: llvm.func @_QPtaskfor()
!LLVMIRDialect:  %[[CONSTANT_2:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[CONSTANT_0:.*]] = llvm.mlir.constant(10 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_1:.*]] = llvm.mlir.constant(1 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_6:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_I:.*]] = llvm.alloca %[[CONSTANT_6]]
!LLVMIRDialect-NEXT:  %[[CONSTANT_8:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_J:.*]] = llvm.alloca %[[CONSTANT_8]]
!LLVMIRDialect-NEXT:  oss.task_for lower_bound(%[[CONSTANT_1]] : i32) upper_bound(%[[CONSTANT_0]] : i32) step(%[[CONSTANT_1]] : i32) loop_type(%[[CONSTANT_2]] : i64) ind_var(%[[VAR_I]] : !llvm.ptr<i32>) private(%[[VAR_I]], %[[VAR_J]] : !llvm.ptr<i32>, !llvm.ptr<i32>)

!LLVMIRDialect-LABEL: llvm.func @_QPtaskloop()
!LLVMIRDialect:  %[[CONSTANT_2:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[CONSTANT_0:.*]] = llvm.mlir.constant(10 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_1:.*]] = llvm.mlir.constant(1 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_6:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_I:.*]] = llvm.alloca %[[CONSTANT_6]]
!LLVMIRDialect-NEXT:  %[[CONSTANT_8:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_J:.*]] = llvm.alloca %[[CONSTANT_8]]
!LLVMIRDialect-NEXT:  oss.taskloop lower_bound(%[[CONSTANT_1]] : i32) upper_bound(%[[CONSTANT_0]] : i32) step(%[[CONSTANT_1]] : i32) loop_type(%[[CONSTANT_2]] : i64) ind_var(%[[VAR_I]] : !llvm.ptr<i32>) private(%[[VAR_I]], %[[VAR_J]] : !llvm.ptr<i32>, !llvm.ptr<i32>)

!LLVMIRDialect-LABEL: llvm.func @_QPtaskloopfor()
!LLVMIRDialect:  %[[CONSTANT_2:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[CONSTANT_0:.*]] = llvm.mlir.constant(10 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_1:.*]] = llvm.mlir.constant(1 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_6:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_I:.*]] = llvm.alloca %[[CONSTANT_6]]
!LLVMIRDialect-NEXT:  %[[CONSTANT_8:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_J:.*]] = llvm.alloca %[[CONSTANT_8]]
!LLVMIRDialect-NEXT:  oss.taskloop_for lower_bound(%[[CONSTANT_1]] : i32) upper_bound(%[[CONSTANT_0]] : i32) step(%[[CONSTANT_1]] : i32) loop_type(%[[CONSTANT_2]] : i64) ind_var(%[[VAR_I]] : !llvm.ptr<i32>) private(%[[VAR_I]], %[[VAR_J]] : !llvm.ptr<i32>, !llvm.ptr<i32>)

