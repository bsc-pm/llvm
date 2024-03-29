! NOTE: Assertions have been autogenerated by /home/rpenacob/llvm-mono/mlir/utils/generate-test-checks.py
! RUN: bbc -hlfir=false -emit-fir %s -o - | FileCheck %s

! Borrowed from Lower/call-copy-in-out.f90

subroutine test_assumed_shape_to_array(x)
  real :: x(:)
  interface
  !$OSS TASK
  subroutine bar(x)
    real, intent(in) :: x(10)
  end subroutine
  end interface

  call bar(x)
end subroutine



! CHECK-LABEL: module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "{{.*}}-unknown-linux-gnu"} {
! CHECK:         func.func @_QPtest_assumed_shape_to_array(%[[VAL_0:[-0-9A-Za-z._]+]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:[-0-9A-Za-z._]+]] = fir.alloca !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_2:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:           %[[VAL_3:[-0-9A-Za-z._]+]] = fir.call @_FortranAIsContiguous(%[[VAL_2]]) fastmath<contract> : (!fir.box<none>) -> i1
! CHECK:           %[[VAL_4:[-0-9A-Za-z._]+]] = fir.if %[[VAL_3]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:             %[[VAL_5:[-0-9A-Za-z._]+]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:             fir.result %[[VAL_5]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           } else {
! CHECK:             %[[VAL_6:[-0-9A-Za-z._]+]] = arith.constant 0 : index
! CHECK:             %[[VAL_7:[-0-9A-Za-z._]+]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:             %[[VAL_8:[-0-9A-Za-z._]+]] = fir.allocmem !fir.array<?xf32>, %[[VAL_7]]#1 {uniq_name = ".copyinout"}
! CHECK:             %[[VAL_9:[-0-9A-Za-z._]+]] = fir.shape %[[VAL_7]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_10:[-0-9A-Za-z._]+]] = fir.embox %[[VAL_8]](%[[VAL_9]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:             fir.store %[[VAL_10]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:             %[[VAL_11:[-0-9A-Za-z._]+]] = fir.address_of(@_QQcl{{.*}})
! CHECK:             %[[VAL_12:[-0-9A-Za-z._]+]] = arith.constant 15 : i32
! CHECK:             %[[VAL_13:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.array<?xf32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_14:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:             %[[VAL_15:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_11]]
! CHECK:             %[[VAL_16:[-0-9A-Za-z._]+]] = fir.call @_FortranAAssignTemporary(%[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_12]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:             fir.result %[[VAL_8]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           }
! CHECK:           %[[VAL_17:[-0-9A-Za-z._]+]] = arith.constant false
! CHECK:           %[[VAL_18:[-0-9A-Za-z._]+]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_17]] : i1
! CHECK:           %[[VAL_19:[-0-9A-Za-z._]+]] = fir.convert %[[VAL_4]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<10xf32>>
! CHECK:           fir.call @_QPbar(%[[VAL_19]]) fastmath<contract> : (!fir.ref<!fir.array<10xf32>>) -> ()
! CHECK:           fir.if %[[VAL_18]] {
! CHECK:             fir.freemem %[[VAL_4]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           }
! CHECK:           return
! CHECK:         }
! CHECK:         func.func private @_QPbar(!fir.ref<!fir.array<10xf32>>)
! CHECK:         func.func private @_FortranAIsContiguous(!fir.box<none>) -> i1 attributes {fir.runtime}
! CHECK:         func.func private @_FortranAAssignTemporary(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none attributes {fir.runtime}
! CHECK:       }

