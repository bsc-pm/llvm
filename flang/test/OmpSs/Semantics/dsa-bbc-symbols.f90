! RUN: bbc -dump-symbols -fompss-2 -o - %s | FileCheck %s

! NOTE: this should output the same as dsa-symbols.f90
! NOTE: At this time pointers and derived types are not implemented yet in flang

program p1
 integer i
 integer array(10)
!$oss task  depend(in:array(i))
 i = i+1
!$oss end task
!$oss task  depend(in:array(i),i)
 array(i) = i+1
!$oss end task
!$oss task  depend(in:array(i)) shared(i)
 array(i) = i+1
!$oss end task
end program

!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   array (OSSShared): HostAssoc
!CHECK:   i (OSSFirstPrivate): HostAssoc
!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   array, OMPSS2_CAPTURE (OSSShared): HostAssoc
!CHECK:   i, OMPSS2_CAPTURE (OSSShared): HostAssoc
!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   array, OMPSS2_CAPTURE (OSSShared): HostAssoc
!CHECK:   i, OMPSS2_CAPTURE (OSSShared): HostAssoc
