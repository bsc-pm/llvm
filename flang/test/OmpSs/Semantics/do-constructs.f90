! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

! This test exists until DO CONCURRENT and FORALL
! FIR emission with OmpSs-2 is fixed

program p
 integer i
 integer j
 integer k(10)
!$oss task
 !ERROR: DO CONCURRENT is not supported yet in OmpSs-2 constructs
 do concurrent(i=1:10)
 end do
!$oss end task
!$oss task
 !ERROR: FORALL is not supported yet in OmpSs-2 constructs
 !ERROR: FORALL is not supported yet in OmpSs-2 constructs
 forall(i=1:10, j=1:10)k(i+j) = 0
!$oss end task
end program

