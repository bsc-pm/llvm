! Test checks directives are taken into account as a lexicalSuccessors.
! In this case STOP must have the OmpSsConstruct as a lexicalSuccessor
! to marks it as a start new block evaluation.

! RUN: bbc -fompss-2 -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

PROGRAM P
    IMPLICIT NONE
    INTEGER ::X

    !$OSS TASK
    X = 888
    !$OSS END TASK

    IF (X /= 1) STOP 1

!    X = 999

    !$OSS TASK
    X = 777
    !$OSS END TASK

END PROGRAM P

!FIRDialect-LABEL: func @_QQmain() {
!FIRDialect:  %[[VAR_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!FIRDialect:  %[[LOAD_VAR_X:.*]] = fir.load %[[VAR_X]] : !fir.ref<i32>
!FIRDialect-NEXT:  %[[CONSTANT_1:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[CMP_0:.*]] = arith.cmpi ne, %[[LOAD_VAR_X]], %[[CONSTANT_1]] : i32
!FIRDialect-NEXT:  cond_br %[[CMP_0]], ^bb1, ^bb2
!FIRDialect: ^bb1:  // pred: ^bb0
!FIRDialect-NEXT:  %[[CONSTANT_1_0:.*]] = arith.constant 1 : i32
!FIRDialect-NEXT:  %[[FALSE_0:.*]] = arith.constant false
!FIRDialect-NEXT:  %[[FALSE_1:.*]] = arith.constant false
!FIRDialect-NEXT:  %[[STOP:.*]] = fir.call @_FortranAStopStatement(%[[CONSTANT_1_0]], %[[FALSE_0]], %[[FALSE_1]]) : (i32, i1, i1) -> none
!FIRDialect-NEXT:  fir.unreachable
!FIRDialect: ^bb2:  // pred: ^bb0
!FIRDialect-NEXT:  oss.task
!FIRDialect-SAME:  {
!FIRDialect-NEXT:    %[[CONSTANT_777:.*]] = arith.constant 777 : i32
!FIRDialect-NEXT:    fir.store %[[CONSTANT_777]] to %[[VAR_X]] : !fir.ref<i32>
!FIRDialect-NEXT:    oss.terminator
!FIRDialect-NEXT:  }

