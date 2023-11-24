! RUN: flang-new -fc1 -emit-fir -fompss-2 -o - %s -flang-deprecated-no-hlfir | FileCheck %s
! XFAIL: true
! NOTE: Fow now let's wait until merge upstream to see what we can do to
! support this

! Tests various aspect of the lowering of polymorphic entities.

! Borrowed from Lower/polymoprhic.f90

module polymorphic_test
  type p1
    integer :: a
    integer :: b
  end type

  contains

  !$OSS TASK
  subroutine pass_poly_pointer_optional(p)
    class(p1), pointer, optional :: p
  end subroutine

  subroutine test_poly_pointer_null()
    call pass_poly_pointer_optional(null())
  end subroutine

end module

