! This test checks lowering of OmpSs-2 task Directive.

! RUN: bbc -fompss-2 %s -o - | \
! RUN:   fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

program task
    INTEGER :: I

    ! No clauses
    !$OSS TASK
    !$OSS END TASK

    ! if clause
    !$OSS TASK IF(I .EQ. 3)
    !$OSS END TASK

    ! final clause
    !$OSS TASK FINAL(I .EQ. 3)
    !$OSS END TASK

    ! cost clause
    !$OSS TASK COST(3)
    !$OSS END TASK

    ! priority clause
    !$OSS TASK PRIORITY(3)
    !$OSS END TASK

    ! default clause
    !$OSS TASK DEFAULT(SHARED)
    !$OSS END TASK

    ! private clause
    !$OSS TASK PRIVATE(I)
    !$OSS END TASK

    ! firstprivate clause
    !$OSS TASK FIRSTPRIVATE(I)
    !$OSS END TASK

    ! shared clause
    !$OSS TASK SHARED(I)
    !$OSS END TASK

    ! In MLIR we can have a branch
    ! whose successors are the same block
    ! with diferent arguments. This is not allowed
    ! in LLVMDialect. Check this
    ! UPDATE: This does not happen anymore, but a
    ! manual mlir code could be written
    ! See mlir/test/Conversion/OmpSsToLLVM/distinc-successors.mlir
    ! !$OSS TASK FIRSTPRIVATE(I)
    !     IF (I .EQ. 1) THEN
    !         I = 1
    !     ELSE IF (I .EQ. 2) THEN
    !         I = 2
    !     ELSE
    !         I = 3
    !     END IF
    ! !$OSS END TASK

end program

!LLVMIRDialect-LABEL: llvm.func @_QQmain()
!LLVMIRDialect:  %[[CONSTANT_3:.*]] = llvm.mlir.constant(3 : i32) : i32
!LLVMIRDialect:  %[[CONSTANT_1:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect:  %[[VAR_I:.*]] = llvm.alloca %[[CONSTANT_1:.*]]

!LLVMIRDialect:  oss.task
!LLVMIRDialect-NEXT:    oss.terminator

!LLVMIRDialect:  %[[LOAD_VAR_I_0:.*]] = llvm.load %[[VAR_I]] : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:  %[[CMP_0:.*]] = llvm.icmp "eq" %[[LOAD_VAR_I_0:.*]], %[[CONSTANT_3]] : i32
!LLVMIRDialect-NEXT:  oss.task if(%[[CMP_0]] : i1)

!LLVMIRDialect:  %[[LOAD_VAR_I_1:.*]] = llvm.load %[[VAR_I]] : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:  %[[CMP_1:.*]] = llvm.icmp "eq" %[[LOAD_VAR_I_1]], %[[CONSTANT_3]] : i32
!LLVMIRDialect-NEXT:  oss.task final(%[[CMP_1]] : i1)

!LLVMIRDialect:  oss.task cost(%[[CONSTANT_3]] : i32)

!LLVMIRDialect:  oss.task priority(%[[CONSTANT_3]] : i32)

!LLVMIRDialect:  oss.task default( defshared)

!LLVMIRDialect:  oss.task private(%[[VAR_I]] : !llvm.ptr<i32>)

!LLVMIRDialect:  oss.task firstprivate(%[[VAR_I]] : !llvm.ptr<i32>)

!LLVMIRDialect:  oss.task shared(%[[VAR_I]] : !llvm.ptr<i32>)
