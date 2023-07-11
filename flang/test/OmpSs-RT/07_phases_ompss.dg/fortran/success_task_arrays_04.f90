! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! test_FFLAGS="--no-copy-deps"
! </testinfo>
module work

contains

subroutine move1(Bout,Ain)
implicit none

real, intent(in)  :: Ain(:,:)
real, intent(out) :: Bout(:,:)

integer :: c1, c2

write(*,*) "bounds1 ",lbound(Bout,1),ubound(Bout,1),lbound(Bout,2),ubound(Bout,2)
do c1=lbound(Bout,1),ubound(Bout,1)
   do c2=lbound(Bout,2),ubound(Bout,2)
      Bout(c1,c2) = Ain(c1,c2) + sin(real(c1))*cos(real(c2))
   end do
end do

return

end subroutine move1

!$OSS TASK IN(Ain) OUT(Bout)
subroutine move2(Bout,Ain)
implicit none

real, intent(in)  :: Ain(:,:)
real, intent(out) :: Bout(:,:)

integer :: c1, c2

write(*,*) "bounds2 ",lbound(Bout,1),ubound(Bout,1),lbound(Bout,2),ubound(Bout,2)
do c1=lbound(Bout,1),ubound(Bout,1)
   do c2=lbound(Bout,2),ubound(Bout,2)
      Bout(c1,c2) = Ain(c1,c2) + sin(real(c1))*cos(real(c2))
   end do
end do

return

end subroutine move2

end module

!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program test

use work
implicit none

real :: a(100,100), b(100,100)
integer :: c1, c2
integer, parameter :: BS = 10

!interface
!   !$OSS TASK IN(Ain) OUT(Bout)
!   subroutine move2(Bout,Ain)
!   implicit none
!   real, intent(in)  :: Ain(:,:)
!   real, intent(out) :: Bout(:,:)
!   end subroutine move2
!end interface

write(*,*) 'start'

call random_number(a)
b = 0.0

do c2=1,100-BS,BS
   do c1=1,100-BS,BS
      call move2(b(c1:c1+BS-1,c2:c2+BS-1),a(c1:c1+BS-1,c2:c2+BS-1))
      !$OSS TASKWAIT
   end do
end do

do c2=1,100-BS,BS
   do c1=1,100-BS,BS
      !$OSS TASK FIRSTPRIVATE(c1, c2) in(a(c1:c1+BS-1,c2:c2+BS-1)) out(b(c1:c1+BS-1,c2:c2+BS-1))
      call move1(b(c1:c1+BS-1,c2:c2+BS-1),a(c1:c1+BS-1,c2:c2+BS-1))
      !$OSS END TASK
      !$OSS TASKWAIT
   end do
end do

write(*,*) 'stop'

stop

end program test
