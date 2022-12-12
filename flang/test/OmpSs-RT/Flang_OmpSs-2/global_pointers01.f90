! RUN: %oss-compile-and-run | FileCheck %s

program pp
   implicit none
   integer, target :: array(10)
   integer, target :: array1(10)
   integer, pointer :: p(:)
   integer, pointer :: p1(:)

   array = 6
   array1 = 7

   p => array
   p1 => array1

   print *, p
   print *, p1

   !$OSS TASK FIRSTPRIVATE(p) PRIVATE(p1)
   print *, ASSOCIATED(p)
   print *, ASSOCIATED(p1)

   print *, p
   p = 4
   print *, p
   nullify(p)
   !$OSS END TASK
   !$OSS TASKWAIT

   print *, p
   print *, p1
end program

! CHECK: 6 6 6 6 6 6 6 6 6 6
! CHECK: 7 7 7 7 7 7 7 7 7 7
! CHECK: T
! CHECK: F
! CHECK: 6 6 6 6 6 6 6 6 6 6
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 4 4 4 4 4 4 4 4 4 4
! CHECK: 7 7 7 7 7 7 7 7 7 7


