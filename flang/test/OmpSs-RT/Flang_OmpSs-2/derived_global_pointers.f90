! RUN: %oss-compile-and-run | FileCheck %s

program pp

   implicit none
   type :: t
     integer, pointer :: s(:)
   end type
   type(t) ty
   integer, target :: s(10)

   s = 7
   ty%s => s

   !$OSS TASK firstprivate(ty)
   print *, associated(ty%s)
   ty%s = 4
   print *, ty%s
   nullify(ty%s)
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty%s

   !$OSS TASK private(ty)
   print *, associated(ty%s)
   nullify(ty%s)
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty%s

   !$OSS TASK shared(ty)
   print *, associated(ty%s)
   ty%s = 9
   print *, ty%s
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty%s

end program

! CHECK: T
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: F
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: T
! CHECK: 9 9 9 9 9 9 9 9 9 9
! CHECK: 9 9 9 9 9 9 9 9 9 9

