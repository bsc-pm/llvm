! This test checks lowering of OmpSs-2 taskloop Directive.

! RUN: bbc -fompss-2 %s -o - | \
! RUN:   fir-opt --cg-rewrite --fir-to-llvm-ir 2>&1 | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect

program task
    INTEGER :: I
    INTEGER :: J

    ! grainsize clause
    !$OSS TASKLOOP GRAINSIZE(60)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! No clauses
    !$OSS TASKLOOP
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! if clause
    !$OSS TASKLOOP IF(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! final clause
    !$OSS TASKLOOP FINAL(I .EQ. 3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! cost clause
    !$OSS TASKLOOP COST(3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! priority clause
    !$OSS TASKLOOP PRIORITY(3)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! default clause
    !$OSS TASKLOOP DEFAULT(SHARED)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! private clause
    !$OSS TASKLOOP PRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! firstprivate clause
    !$OSS TASKLOOP FIRSTPRIVATE(I)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! shared clause
    !$OSS TASKLOOP SHARED(I)
    DO J=1,10
    END DO
    !$OSS END TASKLOOP

    ! In MLIR we can have a branch
    ! whose successors are the same block
    ! with diferent arguments. This is not allowed
    ! in LLVMDialect. Check this
    ! UPDATE: This does not happen anymore, but a
    ! manual mlir code could be written
    ! See mlir/test/Conversion/OmpSsToLLVM/distinc-successors.mlir
    ! !$OSS TASKLOOP PRIVATE(I)
    ! DO J=1,10
    !     IF (I .EQ. 1) THEN
    !         I = 1
    !     ELSE IF (I .EQ. 2) THEN
    !         I = 2
    !     ELSE
    !         I = 3
    !     END IF
    ! END DO
    ! !$OSS END TASKLOOP

end program


!LLVMIRDialect-LABEL: llvm.func @_QQmain()
!LLVMIRDialect:  %[[CONSTANT_3:.*]] = llvm.mlir.constant(3 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_1_0:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[CONSTANT_10:.*]] = llvm.mlir.constant(10 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_1_1:.*]] = llvm.mlir.constant(1 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_60:.*]] = llvm.mlir.constant(60 : i32) : i32
!LLVMIRDialect-NEXT:  %[[CONSTANT_1_2:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_I:.*]] = llvm.alloca %[[CONSTANT_1_2]]
!LLVMIRDialect-NEXT:  %[[CONSTANT_1_3:.*]] = llvm.mlir.constant(1 : i64) : i64
!LLVMIRDialect-NEXT:  %[[VAR_J:.*]] = llvm.alloca %[[CONSTANT_1_3]]
!LLVMIRDialect-NEXT:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) private(%[[VAR_J]] : !llvm.ptr<i32>) grainsize(%[[CONSTANT_60]] : i32)

!LLVMIRDialect:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) private(%[[VAR_J]] : !llvm.ptr<i32>)

!LLVMIRDialect:  %[[LOAD_VAR_I_0:.*]] = llvm.load %[[VAR_I]] : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:  %[[CMP_0:.*]] = llvm.icmp "eq" %[[LOAD_VAR_I_0]], %[[CONSTANT_3]] : i32
!LLVMIRDialect-NEXT:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) if(%[[CMP_0]] : i1) private(%[[VAR_J]] : !llvm.ptr<i32>)

!LLVMIRDialect:  %[[LOAD_VAR_I_1:.*]] = llvm.load %[[VAR_I]] : !llvm.ptr<i32>
!LLVMIRDialect-NEXT:  %[[CMP_1:.*]] = llvm.icmp "eq" %[[LOAD_VAR_I_1]], %[[CONSTANT_3]] : i32
!LLVMIRDialect-NEXT:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) final(%[[CMP_1]] : i1) private(%[[VAR_J]] : !llvm.ptr<i32>)

!LLVMIRDialect:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) cost(%[[CONSTANT_3]] : i32) private(%[[VAR_J]] : !llvm.ptr<i32>)

!LLVMIRDialect:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) priority(%[[CONSTANT_3]] : i32) private(%[[VAR_J]] : !llvm.ptr<i32>)

!LLVMIRDialect:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) private(%[[VAR_J]] : !llvm.ptr<i32>) default( defshared)

!LLVMIRDialect:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) private(%[[VAR_I]], %[[VAR_J]] : !llvm.ptr<i32>, !llvm.ptr<i32>)

!LLVMIRDialect:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) private(%[[VAR_J]] : !llvm.ptr<i32>) firstprivate(%[[VAR_I]] : !llvm.ptr<i32>)

!LLVMIRDialect:  oss.taskloop lower_bound(%[[CONSTANT_1_1]] : i32) upper_bound(%[[CONSTANT_10]] : i32) step(%[[CONSTANT_1_1]] : i32) loop_type(%[[CONSTANT_1_0]] : i64) ind_var(%[[VAR_J]] : !llvm.ptr<i32>) private(%[[VAR_J]] : !llvm.ptr<i32>) shared(%[[VAR_I]] : !llvm.ptr<i32>)
