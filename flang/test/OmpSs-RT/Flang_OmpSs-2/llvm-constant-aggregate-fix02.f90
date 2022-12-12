! RUN: %oss-compile-and-run | FileCheck %s

! This test was not working because
! flang lowers some instruction to a
! ConstantAggregate, and we were not
! handling them. This constant
! was accessing to the original variable

module modType
   implicit none
   type :: t
     integer :: s(10)
     integer :: i(10)
   end type
end module modType

program pp
   use modType
   implicit none
   type(t), allocatable :: ty(:)

   allocate(ty(10))

   ty(1)%s = 7
   ty(1)%i = 7

   !$OSS TASK firstprivate(ty)
   ty(1)%s = 4
   print *, ty(1)%s
   ty(1)%i = 4
   print *, ty(1)%i
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty(1)%s
   print *, ty(1)%i

   !$OSS TASK private(ty)
   print *, allocated(ty)
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty(1)%s
   print *, ty(1)%i

   !$OSS TASK shared(ty)
   ty(1)%s = 10
   print *, ty(1)%s
   ty(1)%i = 10
   print *, ty(1)%i
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty(1)%s
   print *, ty(1)%i

end program

! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: T
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: 10 10 10 10 10 10 10 10 10 10
! CHECK: 10 10 10 10 10 10 10 10 10 10
! CHECK: 10 10 10 10 10 10 10 10 10 10
! CHECK: 10 10 10 10 10 10 10 10 10 10
