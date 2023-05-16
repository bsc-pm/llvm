! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

! Check for assumed-size arrays shared

!DEF: /s (Subroutine) Subprogram
!DEF: /s/x ObjectEntity INTEGER(4)
!DEF: /s/array ObjectEntity INTEGER(4)
subroutine s (x, array)
 !REF: /s/x
 integer x(10)
 !REF: /s/array
 integer array(*)
 !DEF: /s/p PARAMETER ObjectEntity INTEGER(4)
 integer, parameter :: p = 4
!$oss task  depend(in:x(array(p)))
 !DEF: /s/OtherConstruct1/array (OSSShared) HostAssoc INTEGER(4)
 array(2) = 4
!$oss end task
end subroutine

