! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

!DEF: /P1 MainProgram
program P1
 !DEF: /P1/i ObjectEntity INTEGER(4)
 integer i
!$oss task  firstprivate(i)
!$oss task  shared(i)
! Inherits FIRSTPRIVATE from parent
!$oss task
 !DEF: /P1/OtherConstruct1/OtherConstruct1/OtherConstruct1/i (OSSFirstPrivate) HostAssoc INTEGER(4)
 i = 3
!$oss end task
!$oss end task
!$oss end task
!$oss task  default(private)
! Inherits PRIVATE from parent
!$oss task
 !DEF: /P1/OtherConstruct2/OtherConstruct1/i (OSSPrivate) HostAssoc INTEGER(4)
 i = 3
!$oss end task
!$oss end task
!$oss task  default(firstprivate)
! Inherits FIRSTPRIVATE from parent
!$oss task
 !DEF: /P1/OtherConstruct3/OtherConstruct1/i (OSSFirstPrivate) HostAssoc INTEGER(4)
 i = 3
!$oss end task
!$oss end task
!$oss task  shared(i)
! FIRSTPRIVATE of implicit deduction
!$oss task
 !DEF: /P1/OtherConstruct4/OtherConstruct1/i (OSSFirstPrivate) HostAssoc INTEGER(4)
 i = 3
!$oss end task
!$oss end task
end program

