! RUN: bbc -dump-symbols -fompss-2 -o - %s | FileCheck %s

! NOTE: this should output the same as do-symbols.f90
! NOTE: At this moment DO CONCURRENT and FORALL is not supported

program p
 integer i
 integer j
 integer k(10)
!$oss task
 do i=0,10
 end do
!$oss end task
!$oss task
 print *, ((i+j, j=1,10), i=1,10)
!$oss end task
end program

!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   i (OSSPrivate): HostAssoc
!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   i (OSSPrivate): HostAssoc
!CHECK:   j (OSSPrivate): HostAssoc
