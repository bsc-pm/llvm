! RUN: bbc -fompss-2 -polymorphic-type -emit-fir %s -o - | FileCheck %s

! Tests various aspect of the lowering of polymorphic entities.

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

