! NOTE: Assertions have been autogenerated by /home/rpenacob/llvm-mono/mlir/utils/generate-test-checks.py
! This test checks lowering of OmpSs-2 deps.

! explicit-shape non constant shape

! RUN: flang-new -fc1 -emit-fir -fompss-2 -o - %s -flang-deprecated-no-hlfir | FileCheck %s --check-prefix=FIRDialect

PROGRAM P
  IMPLICIT NONE
  INTEGER :: ARRAY(2:10)

  CALL S(2)
  !$OSS TASKWAIT
CONTAINS
SUBROUTINE S(N)
  IMPLICIT NONE
  INTEGER :: N
  INTEGER :: ARRAY(N:10)

  !$OSS TASK IN(ARRAY(4:8))
  !$OSS END TASK
END
END


! FIRDialect-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "p"} {
! FIRDialect:           %[[VAL_0:[-0-9A-Za-z._]+]] = fir.alloca i32 {adapt.valuebyref}
! FIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = arith.constant 2 : i32
! FIRDialect:           fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<i32>
! FIRDialect:           fir.call @_QFPs(%[[VAL_0]]) fastmath<contract> : (!fir.ref<i32>) -> ()
! FIRDialect:           oss.taskwait
! FIRDialect:           return
! FIRDialect:         }

! FIRDialect-LABEL:   func.func private @_QFPs(
! FIRDialect-SAME:                             %[[VAL_0:[-0-9A-Za-z._]+]]: !fir.ref<i32> {fir.bindc_name = "n"})
! FIRDialect:           %[[VAL_1:[-0-9A-Za-z._]+]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! FIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_1]] : (i32) -> i64
! FIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_2]] : (i64) -> index
! FIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = arith.constant 10 : i64
! FIRDialect:           %[[VAL_5:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_4]] : (i64) -> index
! FIRDialect:           %[[VAL_6:[-0-9A-Za-z._]+]] = arith.subi %[[VAL_5]], %[[VAL_3]] : index
! FIRDialect:           %[[VAL_7:[-0-9A-Za-z._]+]] = arith.constant 1 : index
! FIRDialect:           %[[VAL_8:[-0-9A-Za-z._]+]] = arith.addi %[[VAL_6]], %[[VAL_7]] : index
! FIRDialect:           %[[VAL_9:[-0-9A-Za-z._]+]] = arith.constant 0 : index
! FIRDialect:           %[[VAL_10:[-0-9A-Za-z._]+]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : index
! FIRDialect:           %[[VAL_11:[-0-9A-Za-z._]+]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_9]] : index
! FIRDialect:           %[[VAL_12:[-0-9A-Za-z._]+]] = fir.alloca !fir.array<?xi32>, %[[VAL_11]] {bindc_name = "array", uniq_name = "_QFFsEarray"}
! FIRDialect:           %[[VAL_13:[-0-9A-Za-z._]+]] = oss.vlaDim pointer(%[[VAL_12]] : !fir.ref<!fir.array<?xi32>>) sizes(%[[VAL_11]] : index) lbs(%[[VAL_3]] : index) -> i32
! FIRDialect:           %[[VAL_14:[-0-9A-Za-z._]+]] = fir.undefined !fir.oss<!fir.array<?xi32>>
! FIRDialect:           %[[VAL_15:[-0-9A-Za-z._]+]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! FIRDialect:           %[[VAL_16:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_15]] : (i32) -> i64
! FIRDialect:           %[[VAL_17:[-0-9A-Za-z._]+]] = oss.dependency base(%[[VAL_12]] : !fir.ref<!fir.array<?xi32>>) function(@compute.dep0) arguments(%[[VAL_16]], %[[VAL_12]] : i64, !fir.ref<!fir.array<?xi32>>) -> i32
! FIRDialect:           oss.task shared(%[[VAL_12]] : !fir.ref<!fir.array<?xi32>>) shared_type(%[[VAL_14]] : !fir.oss<!fir.array<?xi32>>) vlaDims(%[[VAL_13]] : i32) captures(%[[VAL_11]], %[[VAL_3]], %[[VAL_16]] : index, index, i64) in(%[[VAL_17]] : i32) {
! FIRDialect:             oss.terminator
! FIRDialect:           }
! FIRDialect:           return
! FIRDialect:         }

! FIRDialect-LABEL:   fir.global internal @_QFEarray : !fir.array<9xi32> {
! FIRDialect:           %[[VAL_0:[-0-9A-Za-z._]+]] = fir.zero_bits !fir.array<9xi32>
! FIRDialect:           fir.has_value %[[VAL_0]] : !fir.array<9xi32>
! FIRDialect:         }

! FIRDialect-LABEL:   func.func @compute.dep0(
! FIRDialect-SAME:                            %[[VAL_0:[-0-9A-Za-z._]+]]: i64,
! FIRDialect-SAME:                            %[[VAL_1:[-0-9A-Za-z._]+]]: !fir.ref<!fir.array<?xi32>>) -> (!fir.ref<!fir.array<?xi32>>, i64, i64, i64) {
! FIRDialect:           %[[VAL_2:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_0]] : (i64) -> index
! FIRDialect:           %[[VAL_3:[-0-9A-Za-z._]+]] = arith.constant 10 : i64
! FIRDialect:           %[[VAL_4:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_3]] : (i64) -> index
! FIRDialect:           %[[VAL_5:[-0-9A-Za-z._]+]] = arith.subi %[[VAL_4]], %[[VAL_2]] : index
! FIRDialect:           %[[VAL_6:[-0-9A-Za-z._]+]] = arith.constant 1 : index
! FIRDialect:           %[[VAL_7:[-0-9A-Za-z._]+]] = arith.addi %[[VAL_5]], %[[VAL_6]] : index
! FIRDialect:           %[[VAL_8:[-0-9A-Za-z._]+]] = arith.constant 0 : index
! FIRDialect:           %[[VAL_9:[-0-9A-Za-z._]+]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_8]] : index
! FIRDialect:           %[[VAL_10:[-0-9A-Za-z._]+]] = arith.select %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : index
! FIRDialect:           %[[VAL_11:[-0-9A-Za-z._]+]] = arith.constant 10 : i64
! FIRDialect:           %[[VAL_12:[-0-9A-Za-z._]+]] = arith.subi %[[VAL_11]], %[[VAL_0]] : i64
! FIRDialect:           %[[VAL_13:[-0-9A-Za-z._]+]] = arith.constant 1 : i64
! FIRDialect:           %[[VAL_14:[-0-9A-Za-z._]+]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i64
! FIRDialect:           %[[VAL_15:[-0-9A-Za-z._]+]] = arith.constant 0 : i64
! FIRDialect:           %[[VAL_16:[-0-9A-Za-z._]+]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_15]] : i64
! FIRDialect:           %[[VAL_17:[-0-9A-Za-z._]+]] = arith.select %[[VAL_16]], %[[VAL_14]], %[[VAL_15]] : i64
! FIRDialect:           %[[VAL_18:[-0-9A-Za-z._]+]] = arith.constant 4 : i32
! FIRDialect:           %[[VAL_19:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_18]] : (i32) -> i64
! FIRDialect:           %[[VAL_20:[-0-9A-Za-z._]+]] = arith.constant 8 : i32
! FIRDialect:           %[[VAL_21:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_20]] : (i32) -> i64
! FIRDialect:           %[[VAL_22:[-0-9A-Za-z._]+]] = arith.constant 1 : i64
! FIRDialect:           %[[VAL_23:[-0-9A-Za-z._]+]] = arith.constant 4 : i64
! FIRDialect:           %[[VAL_24:[-0-9A-Za-z._]+]] = arith.subi %[[VAL_19]], %[[VAL_0]] : i64
! FIRDialect:           %[[VAL_25:[-0-9A-Za-z._]+]] = arith.subi %[[VAL_21]], %[[VAL_0]] : i64
! FIRDialect:           %[[VAL_26:[-0-9A-Za-z._]+]] = arith.addi %[[VAL_25]], %[[VAL_22]] : i64
! FIRDialect:           %[[VAL_27:[-0-9A-Za-z._]+]] = arith.muli %[[VAL_17]], %[[VAL_23]] : i64
! FIRDialect:           %[[VAL_28:[-0-9A-Za-z._]+]] = arith.muli %[[VAL_24]], %[[VAL_23]] : i64
! FIRDialect:           %[[VAL_29:[-0-9A-Za-z._]+]] = arith.muli %[[VAL_26]], %[[VAL_23]] : i64
! FIRDialect:           return %[[VAL_1]], %[[VAL_27]], %[[VAL_28]], %[[VAL_29]] : !fir.ref<!fir.array<?xi32>>, i64, i64, i64
! FIRDialect:         }

