! RUN: flang-new -fc1 -emit-fir -fompss-2 -o - %s -flang-deprecated-no-hlfir | FileCheck %s

! The temporary of an actual argument that is not a variable lives in the
! caller frame, so the task copies it.

subroutine s(v)
  integer :: v
  interface
    !$OSS TASK
    subroutine st(x, z)
      integer :: x, z
    end subroutine st
  end interface

  call st(v, 2)
end subroutine s

! CHECK-LABEL: func.func @_QPs(
! CHECK-SAME:      %[[V:.*]]: !fir.ref<i32> {fir.bindc_name = "v"})
! CHECK:         %[[TMP:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK:         oss.task firstprivate(%[[TMP]] : !fir.ref<i32>) {{.*}} shared(%[[V]] : !fir.ref<i32>)
! CHECK-NEXT:      fir.call @_QPst(%[[V]], %[[TMP]])
