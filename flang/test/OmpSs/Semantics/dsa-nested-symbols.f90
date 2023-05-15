! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

!DEF: /p1 MainProgram
program p1
 !DEF: /p1/i OMPSS2_CAPTURE ObjectEntity INTEGER(4)
 integer i
!$oss task  firstprivate(i)
!$oss task  shared(i)
! Inherits FIRSTPRIVATE from parent
!$oss task
 !DEF: /p1/OtherConstruct1/OtherConstruct1/OtherConstruct1/i OMPSS2_CAPTURE (OSSFirstPrivate) HostAssoc INTEGER(4)
 i = 3
!$oss end task
!$oss end task
!$oss end task
!$oss task  default(private)
! Inherits PRIVATE from parent
!$oss task
 !DEF: /p1/OtherConstruct2/OtherConstruct1/i OMPSS2_CAPTURE (OSSPrivate) HostAssoc INTEGER(4)
 i = 3
!$oss end task
!$oss end task
!$oss task  default(firstprivate)
! Inherits FIRSTPRIVATE from parent
!$oss task
 !DEF: /p1/OtherConstruct3/OtherConstruct1/i OMPSS2_CAPTURE (OSSFirstPrivate) HostAssoc INTEGER(4)
 i = 3
!$oss end task
!$oss end task
!$oss task  shared(i)
! FIRSTPRIVATE of implicit deduction
!$oss task
 !DEF: /p1/OtherConstruct4/OtherConstruct1/i OMPSS2_CAPTURE (OSSFirstPrivate) HostAssoc INTEGER(4)
 i = 3
!$oss end task
!$oss end task
end program

