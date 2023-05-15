! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

!DEF: /p1 MainProgram
program p1
 !DEF: /p1/i OMPSS2_CAPTURE ObjectEntity INTEGER(4)
 integer i
!$oss task  default(firstprivate)
 !DEF: /p1/OtherConstruct1/i (OSSFirstPrivate) HostAssoc INTEGER(4)
 i = i+1
 !REF: /p1/OtherConstruct1/i
 i = i+1
!$oss end task 
end program

