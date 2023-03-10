! RUN: %python %S/../../Semantics/test_modfile.py %s %flang_fc1 -fompss-2
! Check modfile generation OmpSs-2 task outline
module m
contains
    !$OSS TASK
    subroutine st()
    end
end

!Expect: m.mod
!module m
! contains
!  !$OSS TASK
!  subroutine st()
!  end
!end

