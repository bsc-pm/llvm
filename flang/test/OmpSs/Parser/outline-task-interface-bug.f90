! RUN: %flang_fc1 -fompss-2 -fdebug-unparse-with-symbols %s

! Until this test we were assuming InterfaceSpecification
! had only InterfaceBody

module posix
    use, intrinsic :: iso_c_binding, only: c_int, c_int32_t
    implicit none

    interface
        ! int usleep(useconds_t useconds)
        function c_usleep(useconds) bind(c, name='usleep')
            import :: c_int, c_int32_t
            integer(kind=c_int32_t), value :: useconds
            integer(kind=c_int)            :: c_usleep
        end function c_usleep
    end interface
end module posix

PROGRAM P
  use :: posix
  IMPLICIT NONE
END PROGRAM
