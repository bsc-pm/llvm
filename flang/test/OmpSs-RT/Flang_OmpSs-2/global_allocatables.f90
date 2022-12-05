! RUN: %oss-compile-and-run | FileCheck %s

program pp
   implicit none
   integer :: array(10)
   integer, allocatable :: s(:)
   integer, allocatable :: fp(:)
   integer, allocatable :: p(:)

   allocate(fp(10))
   allocate(s(10))

   array = 7
   fp = array
   print *, fp

   !$OSS TASK SHARED(s) FIRSTPRIVATE(fp) PRIVATE(p)
   print *, ALLOCATED(s)
   print *, ALLOCATED(fp)
   print *, ALLOCATED(p)
   print *, fp
   fp = 10
   print *, fp
   deallocate(fp)
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, fp
end program

! CHECK:  7 7 7 7 7 7 7 7 7 7
! CHECK:  T
! CHECK:  T
! CHECK:  F
! CHECK:  7 7 7 7 7 7 7 7 7 7
! CHECK:  10 10 10 10 10 10 10 10 10 10
! CHECK:  7 7 7 7 7 7 7 7 7 7
