! RUN: bbc -hlfir=false -pft-test -fompss-2 -o %t %s | FileCheck %s

! Test Pre-FIR Tree captures OmpSs-2 related constructs

! CHECK: Program test_oss
program test_oss
  integer :: i
  ! CHECK: PrintStmt
  print *, "sequential"

  ! CHECK: <<OmpSsConstruct>>
  !$oss task
    ! CHECK: PrintStmt
    print *, "in task"
  !$oss end task
  ! CHECK: <<End OmpSsConstruct>>

  print *, "before taskwait"

  ! CHECK: OmpSsConstruct
  !$oss taskwait

  print *, "before release"

  ! CHECK: OmpSsConstruct
  !$oss release

  print *, "before task do"

  ! CHECK: <<OmpSsConstruct>>
  !$oss task do
  ! CHECK: <<DoConstruct>>
  do i=1,10
    ! CHECK: PrintStmt
    print *, "in task"
  ! CHECK: <<End DoConstruct>>
  end do
  !$oss end task do
  ! CHECK: <<End OmpSsConstruct>>

  print *, "before taskloop"

  ! CHECK: <<OmpSsConstruct>>
  !$oss taskloop
  ! CHECK: <<DoConstruct>>
  do i=1,10
    ! CHECK: PrintStmt
    print *, "in taskloop"
  ! CHECK: <<End DoConstruct>>
  end do
  !$oss end taskloop
  ! CHECK: <<End OmpSsConstruct>>

  print *, "before taskloop do"

  ! CHECK: <<OmpSsConstruct>>
  !$oss taskloop do
  ! CHECK: <<DoConstruct>>
  do i=1,10
    ! CHECK: PrintStmt
    print *, "in taskloop do"
  ! CHECK: <<End DoConstruct>>
  end do
  !$oss end taskloop do
  ! CHECK: <<End OmpSsConstruct>>

  ! CHECK: PrintStmt
  print *, "sequential again"
end program

