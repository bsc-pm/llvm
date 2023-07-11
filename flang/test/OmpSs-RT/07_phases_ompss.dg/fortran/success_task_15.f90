! RUN: %oss-compile-and-run
! XFAIL: true
! <testinfo>
! test_generator=(config/mercurium-ompss "config/mercurium-ompss-2 openmp-compatibility")
! </testinfo>

 program main
 interface
 !$oss task inout(a)
 subroutine sub(a)
         character (len=80) :: a(1:20)
 end subroutine sub
 !$oss task in(a)
 subroutine sub2(a)
         character (len=80) :: a(1:20)
 end subroutine sub2
 end interface

 character (len=80) :: x(1:20)

 x(1) = 'ABCDE'
 x(2) = 'FGHIJ'
 call sub(x)

 call sub2(x)

 !$oss taskwait

 end program main

 subroutine sub(a)
         character (len=80) :: a(1:20)

         a(1) = 'KLMNO'
         a(2) = 'PQRST'
 end subroutine sub

 subroutine sub2(a)
         character (len=80) :: a(1:20)

         IF (a(1) /= 'KLMNO') STOP 1
         IF (a(2) /= 'PQRST') STOP 2
 end subroutine sub2
