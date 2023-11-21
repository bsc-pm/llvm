! This test checks lowering of OmpSs-2 task Directive.

! RUN: bbc -hlfir=false -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

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

end program

!FIRDialect-LABEL: func @_QQmain()
!FIRDialect:  %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
!FIRDialect:  oss.task
!FIRDialect-NEXT:    oss.terminator

!FIRDialect:  %[[LOAD_VAR_I_0:.*]] = fir.load %[[VAR_I]] : !fir.ref<i32>
!FIRDialect-NEXT:  %[[CONSTANT_3_0:.*]] = arith.constant 3 : i32
!FIRDialect-NEXT:  %[[CMP_0:.*]] = arith.cmpi eq, %[[LOAD_VAR_I_0]], %[[CONSTANT_3_0]] : i32
!FIRDialect-NEXT:  oss.task if(%[[CMP_0]] : i1)

!FIRDialect:  %[[LOAD_VAR_I_1:.*]] = fir.load %[[VAR_I]] : !fir.ref<i32>
!FIRDialect-NEXT:  %[[CONSTANT_3_1:.*]] = arith.constant 3 : i32
!FIRDialect-NEXT:  %[[CMP_1:.*]] = arith.cmpi eq, %[[LOAD_VAR_I_1]], %[[CONSTANT_3_1]] : i32
!FIRDialect-NEXT:  oss.task final(%[[CMP_1]] : i1)

!FIRDialect:  %[[CONSTANT_3_2:.*]] = arith.constant 3 : i32
!FIRDialect-NEXT:  oss.task cost(%[[CONSTANT_3_2]] : i32)

!FIRDialect:  %[[CONSTANT_3_3:.*]] = arith.constant 3 : i32
!FIRDialect-NEXT:  oss.task priority(%[[CONSTANT_3_3]] : i32)

!FIRDialect:  oss.task default( defshared)

!FIRDialect:  oss.task private(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect:  oss.task firstprivate(%[[VAR_I]] : !fir.ref<i32>)
!FIRDialect:  oss.task shared(%[[VAR_I]] : !fir.ref<i32>)

