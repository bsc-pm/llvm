! RUN: %oss-compile-and-run | FileCheck %s

program p
 implicit none
 integer, allocatable :: a(:, :)
 integer, allocatable :: b(:, :)

 allocate(a(10, 20))

 !$oss task private(a, b)
   if (.not. allocated(b)) then
     print*, "OK 1!"
   end if
   if (allocated(a)) then
     print*, "OK 2!"
   end if
   if (size(a) == 200) then
     print*, "OK 3!"
   end if
 !$oss end task
 !$oss taskwait

 !$oss task firstprivate(a, b)
   if (.not. allocated(b)) then
     print*, "OK 1!"
   end if
   if (allocated(a)) then
     print*, "OK 2!"
   end if
   if (size(a) == 200) then
     print*, "OK 3!"
   end if
 !$oss end task
 !$oss taskwait
end program p

! CHECK: OK 1!
! CHECK: OK 2!
! CHECK: OK 3!

! CHECK: OK 1!
! CHECK: OK 2!
! CHECK: OK 3!
