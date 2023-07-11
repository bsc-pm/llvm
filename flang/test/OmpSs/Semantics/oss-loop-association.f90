! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

! Check the association between OpenMPLoopConstruct and DoConstruct

  integer :: b = 128
  integer :: c = 32
  integer, parameter :: num = 16
  N = 1024

! Different DO loops

  !$oss taskloop
  do 10 i=1, N
     a = 3.14
10   print *, a
  !$oss end taskloop

  !ERROR: DO CONCURRENT after the TASKLOOP directive is not supported
  !$oss taskloop
  DO CONCURRENT (i = 1:N)
     a = 3.14
  END DO

  !ERROR: DO WHILE after the TASKLOOP directive is not supported
  !$oss taskloop
  outer: DO WHILE (c > 1)
     inner: do while (b > 100)
        a = 3.14
        b = b - 1
     enddo inner
     c = c - 1
  END DO outer

  c = 16
  !ERROR: DO loop after the TASKLOOP directive must have loop control
  !$oss taskloop
  do
     a = 3.14
     c = c - 1
     if (c < 1) exit
  enddo

! Loop association check

  ! If an end do directive follows a do-construct in which several DO
  ! statements share a DO termination statement, then a do directive
  ! can only be specified for the outermost of these DO statements.
  do 100 i=1, N
     !$oss taskloop
     do 100 j=1, N
        a = 3.14
100     continue
    !ERROR: The END TASKLOOP directive must follow the DO loop associated with the loop construct
    !$oss end taskloop

  !$oss taskloop
  do i = 1, N
     !$oss taskloop
     do j = 1, i
     enddo
     !$oss end taskloop
     a = 3.
  enddo
  !$oss end taskloop

  !$oss taskloop
  do i = 1, N
  enddo
  !$oss end taskloop
  !ERROR: The END TASKLOOP directive must follow the DO loop associated with the loop construct
  !$oss end taskloop

  a = 0.0
  !ERROR: The END TASKLOOP directive must follow the DO loop associated with the loop construct
  !$oss end taskloop
  !$oss taskloop
  do i = 1, N
     do j = 1, N
        !ERROR: A DO loop must follow the TASKLOOP directive
        !$oss taskloop
        a = 3.14
     enddo
     !ERROR: The END TASKLOOP directive must follow the DO loop associated with the loop construct
     !$oss end taskloop
  enddo
  a = 1.414
  !ERROR: The END TASKLOOP directive must follow the DO loop associated with the loop construct
  !$oss end taskloop

  do i = 1, N
     !$oss taskloop
     do j = 2*i*N, (2*i+1)*N
        a = 3.14
     enddo
  enddo
  !ERROR: The END TASKLOOP directive must follow the DO loop associated with the loop construct
  !$oss end taskloop

  !ERROR: A DO loop must follow the TASKLOOP directive
  !$oss taskloop
5 FORMAT (1PE12.4, I10)
  do i=1, N
     a = 3.14
  enddo
  !ERROR: The END TASKLOOP directive must follow the DO loop associated with the loop construct
  !$oss end taskloop

  !$oss taskloop
  do i = 1, N
     a = 3.14
  enddo
  !$oss end taskloop
  !ERROR: The END TASKLOOP directive must follow the DO loop associated with the loop construct
  !$oss end taskloop
end
