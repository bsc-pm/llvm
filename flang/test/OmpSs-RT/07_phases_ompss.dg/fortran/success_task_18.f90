! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>

program main
    implicit none
    interface
        !$oss task inout(a)
        subroutine sub(a)
                implicit none
                logical :: a
        end subroutine sub
    end interface

    logical :: b
    b = .false.
    call sub(b)

    !$OSS TASKWAIT

    IF (.NOT. b) STOP 1
end program main

subroutine sub(a)
        implicit none
        logical :: a

        a = .true.
end subroutine sub
