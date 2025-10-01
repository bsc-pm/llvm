! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

!DEF: /P1 MainProgram
program P1
 !DEF: /P1/i ObjectEntity INTEGER(4)
 integer i
 !DEF: /P1/j ObjectEntity INTEGER(4)
 integer j
!$oss task do
 !DEF: /P1/OtherConstruct1/i (OSSPrivate) HostAssoc INTEGER(4)
 do i=1,10
  !DEF: /P1/OtherConstruct1/j (OSSFirstPrivate) HostAssoc INTEGER(4)
  !REF: /P1/OtherConstruct1/i
  j = i
 end do
!$oss end task do
end program

