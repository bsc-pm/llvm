! RUN: %python %S/../../Semantics/test_symbols.py %s %flang_fc1 -fompss-2

!DEF: /P1 MainProgram
program P1
 !DEF: /P1/ty DerivedType
 type :: ty
  !DEF: /P1/ty/array ObjectEntity INTEGER(4)
  integer :: array(10)
 end type
 !DEF: /P1/i ObjectEntity INTEGER(4)
 integer i
 !DEF: /P1/array ObjectEntity INTEGER(4)
 integer array(10)
 !DEF: /P1/p POINTER ObjectEntity INTEGER(4)
 integer, pointer :: p
 !REF: /P1/ty
 !DEF: /P1/t ObjectEntity TYPE(ty)
 type(ty) :: t
!$oss task  depend(in:array(i))
 !DEF: /P1/OtherConstruct1/i (OSSFirstPrivate) HostAssoc INTEGER(4)
 i = i+1
!$oss end task
!$oss task  depend(in:array(i),i)
 !DEF: /P1/OtherConstruct2/array (OSSShared) HostAssoc INTEGER(4)
 !DEF: /P1/OtherConstruct2/i (OSSShared) HostAssoc INTEGER(4)
 array(i) = i+1
!$oss end task
!$oss task  depend(in:array(i)) shared(i)
 !DEF: /P1/OtherConstruct3/array (OSSShared) HostAssoc INTEGER(4)
 !DEF: /P1/OtherConstruct3/i (OSSShared) HostAssoc INTEGER(4)
 array(i) = i+1
!$oss end task
!$oss task  depend(in:p)
 !DEF: /P1/OtherConstruct4/p POINTER (OSSFirstPrivate) HostAssoc INTEGER(4)
 p = 2
!$oss end task
!$oss task  depend(in:t%array(i))
 !DEF: /P1/OtherConstruct5/t (OSSShared) HostAssoc TYPE(ty)
 !REF: /P1/ty/array
 !DEF: /P1/OtherConstruct5/i (OSSFirstPrivate) HostAssoc INTEGER(4)
 t%array(i) = 2
!$oss end task
!$oss task  depend(in:t%array(i),i)
 !DEF: /P1/OtherConstruct6/t (OSSShared) HostAssoc TYPE(ty)
 !REF: /P1/ty/array
 !DEF: /P1/OtherConstruct6/i (OSSShared) HostAssoc INTEGER(4)
 t%array(i) = 2
!$oss end task
!$oss task  depend(in:t%array(i)) shared(i)
 !DEF: /P1/OtherConstruct7/t (OSSShared) HostAssoc TYPE(ty)
 !REF: /P1/ty/array
 !DEF: /P1/OtherConstruct7/i (OSSShared) HostAssoc INTEGER(4)
 t%array(i) = 2
!$oss end task
end program

