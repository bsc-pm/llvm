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
   type(t) ty

   call test(ty)

contains

subroutine test(ty)
   use modType
   implicit none
   type(t) ty

   ty%s = 7
   ty%i = 7

   !$OSS TASK firstprivate(ty)
   ty%s = 4
   print *, ty%s
   ty%i = 4
   print *, ty%i
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty%s
   print *, ty%i

   !$OSS TASK private(ty)
   ty%s = 5
   print *, ty%s
   !ty%i = 5
   !print *, ty%i
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty%s
   print *, ty%i

   !$OSS TASK shared(ty)
   ty%s = 10
   print *, ty%s
   ty%i = 10
   print *, ty%i
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, ty%s
   print *, ty%i

end subroutine
end program

! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: 5 5 5 5 5 5 5 5 5 5
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: 10 10 10 10 10 10 10 10 10 10
! CHECK: 10 10 10 10 10 10 10 10 10 10
! CHECK: 10 10 10 10 10 10 10 10 10 10
! CHECK: 10 10 10 10 10 10 10 10 10 10
