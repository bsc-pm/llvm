! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

!DEF: /p1 MainProgram
program p1
 !DEF: /p1/i OMPSS2_CAPTURE ObjectEntity INTEGER(4)
 integer i
 !DEF: /p1/j OMPSS2_CAPTURE ObjectEntity INTEGER(4)
 integer j
!$oss task do
 !DEF: /p1/OtherConstruct1/i (OSSPrivate) HostAssoc INTEGER(4)
 do i=1,10
  !DEF: /p1/OtherConstruct1/j (OSSFirstPrivate) HostAssoc INTEGER(4)
  !REF: /p1/OtherConstruct1/i
  j = i
 end do
!$oss end task do
end program

