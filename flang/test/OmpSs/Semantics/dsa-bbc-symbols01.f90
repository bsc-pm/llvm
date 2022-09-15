! RUN: bbc -dump-symbols -fompss-2 -o - %s | FileCheck %s

! Check for assumed-size arrays shared
! NOTE: this should output the same as dsa-symbols01.f90

subroutine s (x, array)
 integer x(10)
 integer array(*)
 integer, parameter :: p = 4
!$oss task  depend(in:x(array(p)))
 array(2) = 4
!$oss end task
end subroutine

!CHECK: OtherConstruct scope: size=0 alignment=1
!CHECK:   array (OSSShared): HostAssoc
!CHECK:   x (OSSShared): HostAssoc

