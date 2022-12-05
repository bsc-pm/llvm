! RUN: %oss-compile-and-run

! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>

subroutine fun_task(x, y)
    implicit none
   integer :: x, y(10)

   x = x+ 1
   y(1) = y(1) + 1
end subroutine

subroutine fun_task_2(x, y)
   implicit none
   integer :: x, y(10)

   IF (x /= 42) THEN
       STOP 1
   END IF

   IF (y(1) /= 10042) THEN
       STOP 2
   END IF
end subroutine

program main
    implicit none
    interface

    !$oss task inout(x, y)
    subroutine fun_task(x, y)
       integer :: x
       integer :: y(10)
    end subroutine

    !$oss task in(x, y)
    subroutine fun_task_2(x, y)
       integer :: x
       integer :: y(10)
    end subroutine

    end interface

   integer :: i, j(10)

   i = 41
   j = 10041

   call fun_task(i, j)
   call fun_task_2(i, j)

   !$OSS TASKWAIT

end program
