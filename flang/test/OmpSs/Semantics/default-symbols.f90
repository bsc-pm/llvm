! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

!DEF: /P1 MainProgram
program P1
 !DEF: /P1/i ObjectEntity INTEGER(4)
 integer i
!$oss task  default(firstprivate)
 !DEF: /P1/OtherConstruct1/i (OSSFirstPrivate) HostAssoc INTEGER(4)
 i = i+1
 !REF: /P1/OtherConstruct1/i
 i = i+1
!$oss end task 
end program

