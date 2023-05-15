! RUN: bbc -dump-symbols -fompss-2 -o - %s | FileCheck %s

! NOTE: this should output the same as dsa-nested-symbols.f90

program p1
 integer i
!$oss task  firstprivate(i)
!$oss task  shared(i)
! Inherits FIRSTPRIVATE from parent
!$oss task
 i = 3
!$oss end task
!$oss end task
!$oss end task
!$oss task  default(private)
! Inherits PRIVATE from parent
!$oss task
 i = 3
!$oss end task
!$oss end task
!$oss task  default(firstprivate)
! Inherits FIRSTPRIVATE from parent
!$oss task
 i = 3
!$oss end task
!$oss end task
!$oss task  shared(i)
! FIRSTPRIVATE of implicit deduction
!$oss task
 i = 3
!$oss end task
!$oss end task
end program

!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   i (OSSFirstPrivate): HostAssoc
!CHECK:   OtherConstruct scope: size=0 alignment=1
!CHECK:     i, OMPSS2_CAPTURE (OSSShared): HostAssoc
!CHECK:     OtherConstruct scope: size=0 alignment=1
!CHECK:       i, OMPSS2_CAPTURE (OSSFirstPrivate): HostAssoc
!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   i, OMPSS2_CAPTURE (OSSPrivate): HostAssoc
!CHECK:   OtherConstruct scope: size=0 alignment=1
!CHECK:     i, OMPSS2_CAPTURE (OSSPrivate): HostAssoc
!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   i, OMPSS2_CAPTURE (OSSFirstPrivate): HostAssoc
!CHECK:   OtherConstruct scope: size=0 alignment=1
!CHECK:     i, OMPSS2_CAPTURE (OSSFirstPrivate): HostAssoc
!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   i, OMPSS2_CAPTURE (OSSShared): HostAssoc
!CHECK:   OtherConstruct scope: size=0 alignment=1
!CHECK:     i, OMPSS2_CAPTURE (OSSFirstPrivate): HostAssoc

