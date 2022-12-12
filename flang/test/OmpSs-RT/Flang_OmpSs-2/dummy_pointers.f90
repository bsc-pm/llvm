! RUN: %oss-compile-and-run | FileCheck %s

program pp
   integer, pointer :: s(:)
   integer, pointer :: fp(:)
   integer, pointer :: p(:)

   call test(s, fp, p)

contains
subroutine test(s, fp, p)

   implicit none
   integer, target :: array(10)
   integer, target :: array1(10)
   integer, pointer :: s(:)
   integer, pointer :: fp(:)
   integer, pointer :: p(:)

   array = 6
   array1 = 7

   s => array
   fp => array1

   print *, s
   print *, fp

   !$OSS TASK SHARED(s) FIRSTPRIVATE(fp) PRIVATE(p)
   print *, ASSOCIATED(s)
   print *, ASSOCIATED(fp)
   print *, ASSOCIATED(p)

   print *, s
   print *, fp
   s = 4
   fp = 10
   print *, s
   print *, fp
   nullify(fp)
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, s
   print *, fp
end subroutine test
end program

! CHECK: 6 6 6 6 6 6 6 6 6 6
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: T
! CHECK: T
! CHECK: F
! CHECK: 6 6 6 6 6 6 6 6 6 6
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 10 10 10 10 10 10 10 10 10 10
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 10 10 10 10 10 10 10 10 10 10
