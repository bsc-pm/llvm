! RUN: %oss-compile-and-run | FileCheck %s

program pp
   integer, allocatable :: s(:)
   integer, allocatable :: fp(:)
   integer, allocatable :: p(:)

   call test(s, fp, p)

contains
subroutine test(s, fp, p)

   implicit none
   integer :: array(10)
   integer, allocatable :: s(:)
   integer, allocatable :: fp(:)
   integer, allocatable :: p(:)

   allocate(fp(10))
   allocate(p(10))
   allocate(s(10))

   array = 7
   fp = array
   p = array
   print *, fp

   !$OSS TASK SHARED(s) FIRSTPRIVATE(fp) PRIVATE(p)
   print *, ALLOCATED(s)
   print *, ALLOCATED(fp)
   print *, ALLOCATED(p)
   print *, fp
   p = 0 ! private(p) is allocated but not initialized
   print *, p
   fp = 10
   print *, fp
   deallocate(fp)
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, fp
end subroutine test
end program

! CHECK:  7 7 7 7 7 7 7 7 7 7
! CHECK:  T
! CHECK:  T
! CHECK:  T
! CHECK:  7 7 7 7 7 7 7 7 7 7
! CHECK:  0 0 0 0 0 0 0 0 0 0
! CHECK:  10 10 10 10 10 10 10 10 10 10
! CHECK:  7 7 7 7 7 7 7 7 7 7

