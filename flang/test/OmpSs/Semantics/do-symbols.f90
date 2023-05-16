! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

! Implied-do, FORALL and DO CONCURRENT indices are private.

!DEF: /p MainProgram
program p
 !DEF: /p/i ObjectEntity INTEGER(4)
 integer i
 !DEF: /p/j ObjectEntity INTEGER(4)
 integer j
 !DEF: /p/k ObjectEntity INTEGER(4)
 integer k(10)
!$oss task
 !DEF: /p/OtherConstruct1/i (OSSPrivate) HostAssoc INTEGER(4)
 do i=0,10
 end do
!$oss end task
!$oss task
 !DEF: /p/OtherConstruct2/i (OSSPrivate) HostAssoc INTEGER(4)
 !DEF: /p/OtherConstruct2/j (OSSPrivate) HostAssoc INTEGER(4)
 print *, ((i+j, j=1,10), i=1,10)
!$oss end task
end program


