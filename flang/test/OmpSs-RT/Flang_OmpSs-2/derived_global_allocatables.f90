! RUN: %oss-compile-and-run | FileCheck %s

program pp

   implicit none
   integer :: array(10)
   type :: t
     integer, allocatable :: s(:)
   end type
   type(t) ty

   allocate(ty%s(10))

   ty%s = 7

   !$OSS TASK firstprivate(ty)
   print *, allocated(ty%s)
   ty%s = 4
   print *, ty%s
   deallocate(ty%s)
   !$OSS END TASK
   !$OSS TASKWAIT

   !$OSS TASK private(ty)
   print *, allocated(ty%s)
   deallocate(ty%s)
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty%s

   !$OSS TASK shared(ty)
   print *, allocated(ty%s)
   ty%s = 4
   print *, ty%s
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty%s

end program

! CHECK: T
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: F
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: T
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 4 4 4 4 4 4 4 4 4 4


