! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_FFLAGS="--no-copy-deps"
! </testinfo>

function foo(a)
    implicit none
    integer :: foo
    integer :: a

    foo = a
end function

program p
    implicit none
    integer :: x

    interface
        function foo(a)
            implicit none
            integer :: foo
            integer :: a
        end function
    end interface

    !$oss task priority(10)
    !$oss end task

    !$oss task priority(foo(x))
    !$oss end task

    !$oss task cost(10)
    !$oss end task

    !$oss task cost(foo(x))
    !$oss end task

    !$oss  taskwait
end program p
