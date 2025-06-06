! NOTE: Assertions have been autogenerated by /home/rpenacob/llvm-mono/mlir/utils/generate-test-checks.py
! RUN: flang-new -fc1 -emit-fir -fompss-2 -o - %s -flang-deprecated-no-hlfir | FileCheck %s --check-prefix=FIRDialect

!$OSS TASK OUT(X)
SUBROUTINE S1(X)
  IMPLICIT NONE
  INTEGER :: X
END SUBROUTINE S1

MODULE MOO
CONTAINS
  !$OSS TASK IN(X2)
  SUBROUTINE S2(X2)
    IMPLICIT NONE
    INTEGER :: X2
  END SUBROUTINE S2
END MODULE MOO

PROGRAM MAIN
  USE MOO, ONLY : S2
  IMPLICIT NONE
  INTEGER :: Y
  CALL S1(Y)
  CALL S2(Y)
  !$OSS TASKWAIT
END PROGRAM MAIN 


! FIRDialect-LABEL:   func.func @_QPs1(
! FIRDialect-SAME:                     %[[VAL_0:[-0-9A-Za-z._]+]]: !fir.ref<i32> {fir.bindc_name = "x"}) {
! FIRDialect:           return
! FIRDialect:         }

! FIRDialect-LABEL:   func.func @_QMmooPs2(
! FIRDialect-SAME:                         %[[VAL_0:[-0-9A-Za-z._]+]]: !fir.ref<i32> {fir.bindc_name = "x2"}) {
! FIRDialect:           return
! FIRDialect:         }

! FIRDialect-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "main"} {
! FIRDialect:           %[[VAL_0:[-0-9A-Za-z._]+]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
! FIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = oss.dependency base(%[[VAL_0]] : !fir.ref<i32>) function(@compute.dep0) arguments(%[[VAL_0]] : !fir.ref<i32>) -> i32
! FIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task shared(%[[VAL_0]] : !fir.ref<i32>) shared_type(%[[VAL_2]] : !fir.oss<i32>) out(%[[VAL_1]] : i32) {
! FIRDialect:             fir.call @_QPs1(%[[VAL_0]]) fastmath<contract> : (!fir.ref<i32>) -> ()
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = oss.dependency base(%[[VAL_0]] : !fir.ref<i32>) function(@compute.dep1) arguments(%[[VAL_0]] : !fir.ref<i32>) -> i32
! FIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<i32>
! FIRDialect:           oss.task shared(%[[VAL_0]] : !fir.ref<i32>) shared_type(%[[VAL_4]] : !fir.oss<i32>) in(%[[VAL_3]] : i32) {
! FIRDialect:             fir.call @_QMmooPs2(%[[VAL_0]]) fastmath<contract> : (!fir.ref<i32>) -> ()
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           oss.taskwait
! FIRDialect:           return
! FIRDialect:         }

! FIRDialect-LABEL:   func.func @compute.dep0(
! FIRDialect-SAME:                            %[[VAL_0:[-0-9A-Za-z._]+]]: !fir.ref<i32>) -> (!fir.ref<i32>, i64, i64, i64) {
! FIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = arith.constant 1 : i64
! FIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = arith.constant 1 : i64
! FIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = arith.constant 4 : i64
! FIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = arith.subi %[[VAL_1]], %[[VAL_1]] : i64
! FIRDialect:           %[[VAL_5:[-0-9A-Za-z._]+]] = arith.subi %[[VAL_1]], %[[VAL_1]] : i64
! FIRDialect:           %[[VAL_6:[-0-9A-Za-z._]+]] = arith.addi %[[VAL_5]], %[[VAL_2]] : i64
! FIRDialect:           %[[VAL_7:[-0-9A-Za-z._]+]] = arith.muli %[[VAL_1]], %[[VAL_3]] : i64
! FIRDialect:           %[[VAL_8:[-0-9A-Za-z._]+]] = arith.muli %[[VAL_4]], %[[VAL_3]] : i64
! FIRDialect:           %[[VAL_9:[-0-9A-Za-z._]+]] = arith.muli %[[VAL_6]], %[[VAL_3]] : i64
! FIRDialect:           return %[[VAL_0]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : !fir.ref<i32>, i64, i64, i64
! FIRDialect:         }

! FIRDialect-LABEL:   func.func @compute.dep1(
! FIRDialect-SAME:                            %[[VAL_0:[-0-9A-Za-z._]+]]: !fir.ref<i32>) -> (!fir.ref<i32>, i64, i64, i64) {
! FIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = arith.constant 1 : i64
! FIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = arith.constant 1 : i64
! FIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = arith.constant 4 : i64
! FIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = arith.subi %[[VAL_1]], %[[VAL_1]] : i64
! FIRDialect:           %[[VAL_5:[-0-9A-Za-z._]+]] = arith.subi %[[VAL_1]], %[[VAL_1]] : i64
! FIRDialect:           %[[VAL_6:[-0-9A-Za-z._]+]] = arith.addi %[[VAL_5]], %[[VAL_2]] : i64
! FIRDialect:           %[[VAL_7:[-0-9A-Za-z._]+]] = arith.muli %[[VAL_1]], %[[VAL_3]] : i64
! FIRDialect:           %[[VAL_8:[-0-9A-Za-z._]+]] = arith.muli %[[VAL_4]], %[[VAL_3]] : i64
! FIRDialect:           %[[VAL_9:[-0-9A-Za-z._]+]] = arith.muli %[[VAL_6]], %[[VAL_3]] : i64
! FIRDialect:           return %[[VAL_0]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : !fir.ref<i32>, i64, i64, i64
! FIRDialect:         }

