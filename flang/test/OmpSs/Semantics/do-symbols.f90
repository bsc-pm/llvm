! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

! Implied-do, FORALL and DO CONCURRENT indices are private.

!DEF: /P MainProgram
program P
 !DEF: /P/i ObjectEntity INTEGER(4)
 integer i
 !DEF: /P/j ObjectEntity INTEGER(4)
 integer j
 !DEF: /P/k ObjectEntity INTEGER(4)
 integer k(10)
!$oss task
 !DEF: /P/OtherConstruct1/i (OSSPrivate) HostAssoc INTEGER(4)
 do i=0,10
 end do
!$oss end task
!$oss task
 !DEF: /P/OtherConstruct2/i (OSSPrivate) HostAssoc INTEGER(4)
 !DEF: /P/OtherConstruct2/j (OSSPrivate) HostAssoc INTEGER(4)
 print *, ((i+j, j=1,10), i=1,10)
!$oss end task
end program


